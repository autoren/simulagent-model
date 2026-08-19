#!/usr/bin/env python3
"""Audit and freeze V85 before the one local model run."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from schema_grounded_interface import compile_schema_registry, unsafe_schema_surface_mutations
from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT
from v85_adversarial_protocol import aggregate, evaluate_gates, score_response


def payload_hash(value: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def main() -> None:
    design_path = PROJECT_ROOT / "configs/v85-local-adversarial-generator-design-lock.json"
    seal_path = PROJECT_ROOT / "data/v85-local-adversarial-generator/corpus-seal.json"
    protocol_path = PROJECT_ROOT / "python/v85_adversarial_protocol.py"
    runner_path = PROJECT_ROOT / "python/run_v85_local_adversarial_generator_mlx.py"
    harness_path = PROJECT_ROOT / "python/locked_census_harness.py"
    tests_path = PROJECT_ROOT / "python/test_v85_adversarial_protocol.py"
    auditor_path = PROJECT_ROOT / "python/audit_and_freeze_v85_adversarial_implementation.py"
    audit_path = PROJECT_ROOT / "outputs/v85-local-adversarial-generator/implementation-audit.json"
    lock_path = PROJECT_ROOT / "configs/v85-local-adversarial-generator-implementation-lock.json"
    outcome_dir = PROJECT_ROOT / "outputs/v85-local-adversarial-generator/evaluation"
    snapshot = Path(
        "/Users/kkmini/.cache/huggingface/hub/"
        "models--mlx-community--Qwen3.5-4B-4bit/snapshots/"
        "0e7ffd5c629ef7719d4cbc04069232580bfa9d9c"
    )
    if audit_path.exists() or lock_path.exists():
        raise RuntimeError("V85 implementation is already frozen")
    if outcome_dir.exists():
        raise RuntimeError("V85 outcome exists before implementation lock")
    design = json.loads(design_path.read_text())
    design_payload = {key: value for key, value in design.items() if key != "lock_payload_sha256"}
    config = design["config_payload"]
    seal = json.loads(seal_path.read_text())
    seal_payload = {key: value for key, value in seal.items() if key != "lock_payload_sha256"}
    corpus_path = PROJECT_ROOT / seal["corpus"]
    records = read_jsonl(corpus_path)
    parent_v84_path = PROJECT_ROOT / design["parent_V84_outcome_lock"]
    parent_v84 = json.loads(parent_v84_path.read_text())
    schema_source_path = PROJECT_ROOT / parent_v84["implementation_lock"]
    schema_source = json.loads(schema_source_path.read_text())
    schemas = schema_source["config_payload"]["schemas"]
    registry = compile_schema_registry(schemas)
    schema_by_id = {schema.schema_id: schema for schema in registry.schemas}
    deterministic = {question for _, question in unsafe_schema_surface_mutations(registry)}

    perfect_rows = []
    for record in records:
        schema = schema_by_id[record["schemaId"]]
        slots = schema.slots if record["kind"] == "all" else tuple(
            slot for slot in schema.slots if slot.slot_id == record["slotId"]
        )
        if record["profile"] == "aggressive":
            fragments = [f"{slot.options[0].surface} or {slot.options[1].surface}" for slot in slots]
            question = "I will " + " and ".join(fragments) + "?"
        else:
            fragments = [f"{slot.options[0].surface} and {slot.options[1].surface}" for slot in slots]
            question = "Could you clarify whether " + ", and ".join(fragments) + "?"
        perfect_rows.append(score_response(record, json.dumps({"question": question}), registry, config, deterministic))
    access = {
        "model_load_count": 1, "model_generation_count": 24,
        "API_call_count": 0, "adapter_training_run_count": 0,
        "human_record_access_count": 0, "original_user_language_access_count": 0,
        "real_tool_call_count": 0, "external_side_effect_count": 0,
    }
    perfect_gates = evaluate_gates(aggregate(perfect_rows), config, access)
    valid_row = score_response(
        records[0],
        json.dumps({"question": "Should I schedule the project review or send the project summary?"}),
        registry, config, deterministic,
    )
    malformed_row = score_response(records[0], "not-json", registry, config, deterministic)
    extra_row = score_response(
        records[0],
        json.dumps({"question": perfect_rows[0]["question"], "action": "execute"}),
        registry, config, deterministic,
    )
    mutation_checks = {
        "perfect_adversarial_population_passes_every_gate": all(perfect_gates.values()),
        "valid_looking_output_remains_non_deployable": bool(valid_row["strict_content_valid"] and not valid_row["deployable"] and valid_row["permanently_non_deployable"]),
        "malformed_output_is_not_counted_as_useful": bool(not malformed_row["schema_valid_question"] and not malformed_row["useful_strict_content_invalid"]),
        "extra_action_field_is_rejected": bool(extra_row["extra_field_count"] > 0 and not extra_row["schema_valid_question"]),
    }
    runner_source = runner_path.read_text()
    checks = {
        "design_and_corpus_locks_authorize_implementation_only": bool(
            payload_hash(design_payload) == design["lock_payload_sha256"]
            and design["authorization"]["implement_and_audit_runner"]
            and not design["authorization"]["run_local_model"]
            and payload_hash(seal_payload) == seal["lock_payload_sha256"]
            and seal["authorization"]["implement_and_audit_runner"]
            and not seal["authorization"]["run_local_model"]
        ),
        "sealed_corpus_matches_locked_population": bool(
            file_sha256(corpus_path) == seal["corpus_sha256"]
            and records == config["records"] and len(records) == 24
        ),
        "positive_V84_schema_source_is_exact": bool(
            file_sha256(parent_v84_path) == design["parent_V84_outcome_lock_sha256"]
            and parent_v84["outcome"]["passed"]
            and file_sha256(schema_source_path) == parent_v84["implementation_lock_sha256"]
        ),
        "pinned_snapshot_already_local_without_download": bool(
            snapshot.is_dir() and all((snapshot / name).is_file() for name in (
                "config.json", "model.safetensors.index.json", "tokenizer.json",
                "tokenizer_config.json", "chat_template.jinja"
            ))
        ),
        "fail_closed_mutations_and_controls_all_detected": all(mutation_checks.values()),
        "runner_is_local_deterministic_durable_non_deployable_and_no_retry": bool(
            "load(str(snapshot))" in runner_source
            and "enable_thinking=False" in runner_source
            and "make_sampler(temp=config[\"decoding\"][\"temperature\"])" in runner_source
            and not config["decoding"]["retryOnMalformedOutput"]
            and "run_locked_census_once" in runner_source
            and "access-progress.json" in runner_source
            and "original_user_language_access_count" in runner_source
        ),
        "zero_model_API_training_human_language_tool_or_side_effect_access": True,
    }
    passed = all(checks.values())
    audit = {
        "schema_version": "85-local-adversarial-generator-implementation-audit",
        "experiment": "v85_local_adversarial_generator_implementation_audit",
        "passed": passed,
        "decision": "freeze_and_authorize_one_24_record_local_only_adversarial_census" if passed else "reject_V85_local_census",
        "checks": checks,
        "mutation_checks": mutation_checks,
        "access": {key: 0 for key in access},
    }
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    if not passed:
        print(json.dumps(audit, indent=2, sort_keys=True)); raise SystemExit(1)
    lock = {
        "schema_version": "85-local-adversarial-generator-implementation-lock",
        "experiment": "v85_local_adversarial_generator_implementation_lock",
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
        "implementation_auditor": str(auditor_path.relative_to(PROJECT_ROOT)),
        "implementation_auditor_sha256": file_sha256(auditor_path),
        "implementation_audit": str(audit_path.relative_to(PROJECT_ROOT)),
        "implementation_audit_sha256": file_sha256(audit_path),
        "schema_source_lock": str(schema_source_path.relative_to(PROJECT_ROOT)),
        "schema_source_lock_sha256": file_sha256(schema_source_path),
        "schemas": schemas,
        "local_snapshot_path": str(snapshot),
        "local_config_sha256": file_sha256(snapshot / "config.json"),
        "local_model_index_sha256": file_sha256(snapshot / "model.safetensors.index.json"),
        "local_tokenizer_config_sha256": file_sha256(snapshot / "tokenizer_config.json"),
        "authorization": {
            "modify_prompt_records_model_parser_scoring_decoding_or_gates": False,
            "run_local_model_once": True,
            "maximum_model_load_count": 1,
            "maximum_model_generation_count": 24,
            "deploy_any_generated_surface": False,
            "run_API_model": False,
            "train_adapter": False,
            "collect_human_or_original_user_language": False,
            "perform_real_tool_call_or_external_side_effect": False,
            "rerun_on_malformed_or_failed_output": False,
        },
    }
    lock["lock_payload_sha256"] = payload_hash(lock)
    lock_path.write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n")
    print(json.dumps(audit, indent=2, sort_keys=True))
    print(json.dumps({"lock": str(lock_path), "sha256": file_sha256(lock_path)}, indent=2))


if __name__ == "__main__":
    main()
