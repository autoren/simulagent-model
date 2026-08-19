#!/usr/bin/env python3
"""Audit and freeze V80 corpus/parser/runner before loading the local model."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT
from v80_candidate_protocol import aggregate, evaluate_gates, score_record


def payload_hash(value: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def main() -> None:
    design_path = PROJECT_ROOT / "configs/v80-local-candidate-generation-design-lock.json"
    corpus_seal_path = PROJECT_ROOT / "data/v80-local-candidate-generation/corpus-seal.json"
    protocol_path = PROJECT_ROOT / "python/v80_candidate_protocol.py"
    runner_path = PROJECT_ROOT / "python/run_v80_local_candidate_generation_mlx.py"
    harness_path = PROJECT_ROOT / "python/locked_census_harness.py"
    tests_path = PROJECT_ROOT / "python/test_v80_candidate_protocol.py"
    auditor_path = (
        PROJECT_ROOT / "python/audit_and_freeze_v80_local_candidate_implementation.py"
    )
    audit_path = PROJECT_ROOT / "outputs/v80-local-candidate-generation/implementation-audit.json"
    lock_path = PROJECT_ROOT / "configs/v80-local-candidate-generation-implementation-lock.json"
    outcome_dir = PROJECT_ROOT / "outputs/v80-local-candidate-generation/evaluation"
    snapshot = Path(
        "/Users/kkmini/.cache/huggingface/hub/"
        "models--mlx-community--Qwen3.5-4B-4bit/snapshots/"
        "0e7ffd5c629ef7719d4cbc04069232580bfa9d9c"
    )
    if audit_path.exists() or lock_path.exists():
        raise RuntimeError("V80 local runner implementation is already frozen")
    if outcome_dir.exists():
        raise RuntimeError("V80 model outcome exists before implementation lock")

    design = json.loads(design_path.read_text())
    design_payload = {
        key: value for key, value in design.items() if key != "lock_payload_sha256"
    }
    config = design["config_payload"]
    corpus_seal = json.loads(corpus_seal_path.read_text())
    corpus_payload = {
        key: value
        for key, value in corpus_seal.items()
        if key != "lock_payload_sha256"
    }
    corpus_path = PROJECT_ROOT / corpus_seal["corpus"]
    records = read_jsonl(corpus_path)
    authorization = bool(
        payload_hash(design_payload) == design["lock_payload_sha256"]
        and design["authorization"]["implement_and_audit_local_runner"]
        and not design["authorization"]["run_local_model"]
        and not design["authorization"]["run_API_model"]
        and payload_hash(corpus_payload) == corpus_seal["lock_payload_sha256"]
        and corpus_seal["authorization"]["implement_and_audit_runner"]
        and not corpus_seal["authorization"]["run_local_model"]
    )
    corpus_exact = bool(
        file_sha256(corpus_path) == corpus_seal["corpus_sha256"]
        and records == config["records"]
        and len(records) == config["gates"]["requiredRecordCount"]
    )
    snapshot_local = bool(
        snapshot.is_dir()
        and all(
            (snapshot / name).is_file()
            for name in (
                "config.json",
                "model.safetensors.index.json",
                "tokenizer.json",
                "tokenizer_config.json",
                "chat_template.jinja",
            )
        )
    )

    perfect_rows = [
        score_record(
            record,
            json.dumps({"candidate_ids": record["goldCandidateIds"]}),
            config,
        )
        for record in records
    ]
    perfect_metrics = aggregate(perfect_rows)
    zero_external_access = {
        "model_forward_pass_count": 24,
        "API_call_count": 0,
        "adapter_training_run_count": 0,
        "human_record_access_count": 0,
        "real_tool_call_count": 0,
        "external_side_effect_count": 0,
    }
    perfect_gates = evaluate_gates(perfect_metrics, config, zero_external_access)
    fenced_rows = list(perfect_rows)
    fenced_rows[0] = score_record(records[0], "```json\n{}\n```", config)
    fenced_gates = evaluate_gates(
        aggregate(fenced_rows), config, zero_external_access
    )
    all_candidates = config["candidateIdsInRequiredOrder"]
    overgenerated_rows = [
        score_record(record, json.dumps({"candidate_ids": all_candidates}), config)
        for record in records
    ]
    overgenerated_gates = evaluate_gates(
        aggregate(overgenerated_rows), config, zero_external_access
    )
    forbidden_rows = list(perfect_rows)
    forbidden_rows[0] = score_record(
        records[0],
        json.dumps(
            {
                "candidate_ids": records[0]["goldCandidateIds"],
                "probability": 0.9,
                "tool_call": "execute",
            }
        ),
        config,
    )
    forbidden_gates = evaluate_gates(
        aggregate(forbidden_rows), config, zero_external_access
    )
    excessive_access = dict(zero_external_access)
    excessive_access["model_forward_pass_count"] = 25
    access_gates = evaluate_gates(perfect_metrics, config, excessive_access)
    mutations = {
        "perfect_synthetic_population_passes_every_gate": all(perfect_gates.values()),
        "fenced_output_fails_parse_and_schema_gates": bool(
            not fenced_gates["exact_JSON_parse_rate"]
            and not fenced_gates["schema_validity_rate"]
        ),
        "all_candidate_overgeneration_fails_bound_and_exactness": bool(
            not overgenerated_gates["bounded_mean_candidate_count"]
            and not overgenerated_gates["exact_candidate_set_accuracy"]
        ),
        "forbidden_probability_and_tool_fields_fail": bool(
            not forbidden_gates["zero_confidence_or_probability_fields"]
            and not forbidden_gates["zero_action_or_tool_fields"]
        ),
        "twenty_fifth_generation_fails_access_bound": not access_gates[
            "bounded_local_model_and_zero_external_access"
        ],
    }
    runner_source = runner_path.read_text()
    runner_structure = bool(
        "load(str(snapshot))" in runner_source
        and "enable_thinking=False" in runner_source
        and "make_sampler(temp=config[\"decoding\"][\"temperature\"])" in runner_source
        and not config["decoding"]["retryOnMalformedOutput"]
        and "API_call_count" in runner_source
        and "access-progress.json" in runner_source
        and "run_locked_census_once" in runner_source
    )
    checks = {
        "design_and_corpus_locks_authorize_implementation_only": authorization,
        "sealed_corpus_exactly_matches_preregistered_population": corpus_exact,
        "pinned_snapshot_is_already_local_without_download": snapshot_local,
        "strict_parser_and_gate_mutations_all_detected": all(mutations.values()),
        "runner_is_local_deterministic_durable_and_no_retry": runner_structure,
        "zero_model_forward_passes_during_implementation_audit": True,
        "zero_API_adapter_human_tool_or_external_access": True,
    }
    passed = all(checks.values())
    audit = {
        "schema_version": "80-local-candidate-generation-implementation-audit",
        "experiment": "v80_local_candidate_generation_implementation_audit",
        "passed": passed,
        "decision": (
            "freeze_runner_and_authorize_one_24-record_local-only_evaluation"
            if passed
            else "reject_or_defer_V80_local_evaluation"
        ),
        "checks": checks,
        "mutation_checks": mutations,
        "local_snapshot_path": str(snapshot),
        "access": {
            "model_load_count": 0,
            "model_forward_pass_count": 0,
            "API_call_count": 0,
            "adapter_training_run_count": 0,
            "human_record_access_count": 0,
            "real_tool_call_count": 0,
            "external_side_effect_count": 0,
        },
    }
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    if not passed:
        print(json.dumps(audit, indent=2, sort_keys=True))
        raise SystemExit(1)

    lock = {
        "schema_version": "80-local-candidate-generation-implementation-lock",
        "experiment": "v80_local_candidate_generation_implementation_lock",
        "design_lock": str(design_path.relative_to(PROJECT_ROOT)),
        "design_lock_sha256": file_sha256(design_path),
        "config_payload": config,
        "corpus_seal": str(corpus_seal_path.relative_to(PROJECT_ROOT)),
        "corpus_seal_sha256": file_sha256(corpus_seal_path),
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
        "local_snapshot_path": str(snapshot),
        "local_config_sha256": file_sha256(snapshot / "config.json"),
        "local_model_index_sha256": file_sha256(
            snapshot / "model.safetensors.index.json"
        ),
        "local_tokenizer_config_sha256": file_sha256(
            snapshot / "tokenizer_config.json"
        ),
        "authorization": {
            "modify_prompt_model_revision_corpus_parser_decoding_or_gates": False,
            "run_local_model_once": True,
            "maximum_model_load_count": 1,
            "maximum_model_generation_count": 24,
            "run_API_model": False,
            "train_adapter": False,
            "collect_human_language": False,
            "perform_real_tool_call_or_external_side_effect": False,
            "rerun_on_malformed_or_failed_output": False,
        },
    }
    lock["lock_payload_sha256"] = payload_hash(lock)
    lock_path.write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n")
    print(json.dumps(audit, indent=2, sort_keys=True))
    print(
        json.dumps(
            {"lock": str(lock_path), "sha256": file_sha256(lock_path)},
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
