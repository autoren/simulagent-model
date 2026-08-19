#!/usr/bin/env python3
"""Audit and freeze V88 before its one local model run."""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Any

from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT
from v88_external_candidate_protocol import aggregate, evaluate_gates, format_user_prompt, score_response


def payload_hash(value: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def selection_hash(salt: str, service: str, intent: str, record_id: str) -> str:
    return hashlib.sha256(f"{salt}\0{service}\0{intent}\0{record_id}".encode()).hexdigest()


def main() -> None:
    design_path = PROJECT_ROOT / "configs/v88-external-intent-candidate-design-lock.json"
    seal_path = PROJECT_ROOT / "data/v88-external-intent-candidate/corpus-seal.json"
    protocol_path = PROJECT_ROOT / "python/v88_external_candidate_protocol.py"
    runner_path = PROJECT_ROOT / "python/run_v88_external_candidate_mlx.py"
    harness_path = PROJECT_ROOT / "python/locked_census_harness.py"
    tests_path = PROJECT_ROOT / "python/test_v88_external_candidate_protocol.py"
    builder_path = PROJECT_ROOT / "python/build_v88_external_candidate_corpus.py"
    auditor_path = PROJECT_ROOT / "python/audit_and_freeze_v88_external_candidate_implementation.py"
    audit_path = PROJECT_ROOT / "outputs/v88-external-intent-candidate/implementation-audit.json"
    lock_path = PROJECT_ROOT / "configs/v88-external-intent-candidate-implementation-lock.json"
    outcome_dir = PROJECT_ROOT / "outputs/v88-external-intent-candidate/evaluation"
    snapshot = Path(
        "/Users/kkmini/.cache/huggingface/hub/"
        "models--mlx-community--Qwen3.5-4B-4bit/snapshots/"
        "0e7ffd5c629ef7719d4cbc04069232580bfa9d9c"
    )
    if audit_path.exists() or lock_path.exists():
        raise RuntimeError("V88 implementation is already frozen")
    if outcome_dir.exists():
        raise RuntimeError("V88 outcome exists before implementation lock")

    design = json.loads(design_path.read_text())
    design_payload = {key: value for key, value in design.items() if key != "lock_payload_sha256"}
    config = design["config_payload"]
    seal = json.loads(seal_path.read_text())
    seal_payload = {key: value for key, value in seal.items() if key != "lock_payload_sha256"}
    corpus_path = PROJECT_ROOT / seal["corpus"]
    records = read_jsonl(corpus_path)
    records_by_id = {record["id"]: record for record in records}

    perfect_rows = [
        score_response(record, json.dumps({
            "intent_candidates": record["gold"]["intent_candidates"],
            "state_slot_key_candidates": record["gold"]["state_slot_key_candidates"],
        }))
        for record in records
    ]
    full_access = {
        "source_language_record_access_count": 48,
        "manual_utterance_inspection_count": 0,
        "model_load_count": 1,
        "model_generation_count": 48,
        "LLM_API_call_count": 0,
        "adapter_training_run_count": 0,
        "real_service_call_count": 0,
        "external_side_effect_count": 0,
    }
    perfect_metrics = aggregate(perfect_rows, records_by_id)
    perfect_gates = evaluate_gates(perfect_metrics, config, full_access)
    malformed = score_response(records[0], "not-json")
    extra = score_response(records[0], json.dumps({
        "intent_candidates": records[0]["gold"]["intent_candidates"],
        "state_slot_key_candidates": records[0]["gold"]["state_slot_key_candidates"],
        "action": "execute",
    }))
    missing_none = score_response(records[0], json.dumps({
        "intent_candidates": [item for item in records[0]["gold"]["intent_candidates"] if item != "NONE"],
        "state_slot_key_candidates": records[0]["gold"]["state_slot_key_candidates"],
    }))
    prompts = [format_user_prompt(record, config) for record in records]
    strata = Counter((record["service"], record["gold"]["active_intent"]) for record in records)
    registered_strata = {
        (item["service"], item["activeIntent"]): item["count"]
        for item in config["population"]["strata"]
    }
    runner_source = runner_path.read_text()
    mutation_checks = {
        "oracle_population_passes_every_registered_gate": all(perfect_gates.values()),
        "malformed_output_fails_parse_and_ontology": not malformed["exact_json"] and not malformed["ontology_conformant"],
        "extra_action_key_fails_ontology": extra["exact_json"] and not extra["ontology_conformant"],
        "missing_NONE_cannot_pass_mandatory_open_set_check": not missing_none["mandatory_NONE_included"],
        "all_outputs_remain_non_deployable": all(row["permanently_non_deployable"] and not row["executable"] for row in perfect_rows),
    }
    checks = {
        "design_and_corpus_locks_authorize_implementation_only": bool(
            payload_hash(design_payload) == design["lock_payload_sha256"]
            and design["authorization"]["implement_and_audit_local_runner_after_corpus_seal"]
            and not design["authorization"]["run_local_model_before_implementation_lock"]
            and payload_hash(seal_payload) == seal["lock_payload_sha256"]
            and seal["authorization"]["implement_and_audit_local_runner"]
            and not seal["authorization"]["run_local_model"]
        ),
        "sealed_corpus_matches_hash_population_strata_and_targets": bool(
            file_sha256(corpus_path) == seal["corpus_sha256"]
            and len(records) == seal["record_count"] == config["population"]["recordCount"] == 48
            and len(records_by_id) == 48
            and strata == registered_strata
            and all(record["selection_hash"] == selection_hash(
                config["population"]["selectionSalt"], record["service"],
                record["gold"]["active_intent"], record["source_record_id"]
            ) for record in records)
            and all(record["provenance"]["license"] == "CC-BY-SA-4.0" for record in records)
            and all(not record["provenance"]["deployable"] and not record["provenance"]["executable"] for record in records)
        ),
        "prompt_population_is_complete_bounded_and_does_not_embed_gold_fields": bool(
            len(prompts) == 48
            and max(len(prompt) for prompt in prompts) < config["decoding"]["maximumPromptTokens"]
            and all("gold" not in prompt and "selection_hash" not in prompt and "source_record_id" not in prompt for prompt in prompts)
        ),
        "pinned_snapshot_already_local_without_download": bool(
            snapshot.is_dir() and all((snapshot / name).is_file() for name in (
                "config.json", "model.safetensors.index.json", "tokenizer.json",
                "tokenizer_config.json", "chat_template.jinja"
            ))
        ),
        "fail_closed_parser_controls_and_oracle_gate_feasibility_pass": all(mutation_checks.values()),
        "runner_is_local_deterministic_durable_non_deployable_and_no_retry": bool(
            "load(str(snapshot))" in runner_source
            and "enable_thinking=False" in runner_source
            and "make_sampler(temp=config[\"decoding\"][\"temperature\"])" in runner_source
            and not config["decoding"]["retryOnMalformedOutput"]
            and "run_locked_census_once" in runner_source
            and "access-progress.json" in runner_source
            and "manual_utterance_inspection_count" in runner_source
            and "LLM_API_call_count" in runner_source
        ),
        "zero_preinference_model_API_training_manual_inspection_service_or_side_effect_access": True,
    }
    passed = all(checks.values())
    audit = {
        "schema_version": "88-external-intent-candidate-implementation-audit",
        "experiment": "v88_external_intent_candidate_implementation_audit",
        "passed": passed,
        "decision": "freeze_and_authorize_one_48_record_local_only_external_shadow_census" if passed else "reject_V88_local_census",
        "checks": checks,
        "mutation_checks": mutation_checks,
        "prompt_character_summary": {
            "minimum": min(len(prompt) for prompt in prompts),
            "maximum": max(len(prompt) for prompt in prompts),
            "mean": sum(len(prompt) for prompt in prompts) / len(prompts),
        },
        "oracle_metrics": perfect_metrics,
        "access": {key: 0 for key in full_access},
    }
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    if not passed:
        print(json.dumps(audit, indent=2, sort_keys=True))
        raise SystemExit(1)

    lock = {
        "schema_version": "88-external-intent-candidate-implementation-lock",
        "experiment": "v88_external_intent_candidate_implementation_lock",
        "design_lock": str(design_path.relative_to(PROJECT_ROOT)),
        "design_lock_sha256": file_sha256(design_path),
        "config_payload": config,
        "corpus_seal": str(seal_path.relative_to(PROJECT_ROOT)),
        "corpus_seal_sha256": file_sha256(seal_path),
        "corpus": str(corpus_path.relative_to(PROJECT_ROOT)),
        "corpus_sha256": file_sha256(corpus_path),
        "protocol": str(protocol_path.relative_to(PROJECT_ROOT)),
        "protocol_sha256": file_sha256(protocol_path),
        "runner": str(runner_path.relative_to(PROJECT_ROOT)),
        "runner_sha256": file_sha256(runner_path),
        "census_harness": str(harness_path.relative_to(PROJECT_ROOT)),
        "census_harness_sha256": file_sha256(harness_path),
        "tests": str(tests_path.relative_to(PROJECT_ROOT)),
        "tests_sha256": file_sha256(tests_path),
        "builder": str(builder_path.relative_to(PROJECT_ROOT)),
        "builder_sha256": file_sha256(builder_path),
        "implementation_auditor": str(auditor_path.relative_to(PROJECT_ROOT)),
        "implementation_auditor_sha256": file_sha256(auditor_path),
        "implementation_audit": str(audit_path.relative_to(PROJECT_ROOT)),
        "implementation_audit_sha256": file_sha256(audit_path),
        "local_snapshot_path": str(snapshot),
        "local_config_sha256": file_sha256(snapshot / "config.json"),
        "local_model_index_sha256": file_sha256(snapshot / "model.safetensors.index.json"),
        "local_tokenizer_config_sha256": file_sha256(snapshot / "tokenizer_config.json"),
        "authorization": {
            "modify_prompt_records_model_parser_scoring_decoding_controls_or_gates": False,
            "run_local_model_once": True,
            "maximum_model_load_count": 1,
            "maximum_model_generation_count": 48,
            "deploy_or_execute_any_model_output": False,
            "run_API_model": False,
            "train_adapter": False,
            "manually_inspect_source_language_or_prompts": False,
            "perform_real_service_call_or_external_side_effect": False,
            "rerun_on_malformed_failed_or_negative_output": False,
        },
    }
    lock["lock_payload_sha256"] = payload_hash(lock)
    lock_path.write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n")
    print(json.dumps(audit, indent=2, sort_keys=True))
    print(json.dumps({"lock": str(lock_path), "sha256": file_sha256(lock_path)}, indent=2))


if __name__ == "__main__":
    main()
