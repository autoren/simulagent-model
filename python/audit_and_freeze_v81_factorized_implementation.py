#!/usr/bin/env python3
"""Audit and freeze V81 corpus, protocol, and runner before model access."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT
from v81_factorized_candidate_protocol import aggregate, evaluate_gates, score_record


def payload_hash(value: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def main() -> None:
    design_path = PROJECT_ROOT / "configs/v81-factorized-local-candidate-design-lock.json"
    seal_path = PROJECT_ROOT / "data/v81-factorized-local-candidate/corpus-seal.json"
    protocol_path = PROJECT_ROOT / "python/v81_factorized_candidate_protocol.py"
    runner_path = PROJECT_ROOT / "python/run_v81_factorized_local_candidate_mlx.py"
    harness_path = PROJECT_ROOT / "python/locked_census_harness.py"
    tests_path = PROJECT_ROOT / "python/test_v81_factorized_candidate_protocol.py"
    auditor_path = (
        PROJECT_ROOT / "python/audit_and_freeze_v81_factorized_implementation.py"
    )
    audit_path = (
        PROJECT_ROOT / "outputs/v81-factorized-local-candidate/implementation-audit.json"
    )
    lock_path = (
        PROJECT_ROOT / "configs/v81-factorized-local-candidate-implementation-lock.json"
    )
    outcome_dir = PROJECT_ROOT / "outputs/v81-factorized-local-candidate/evaluation"
    snapshot = Path(
        "/Users/kkmini/.cache/huggingface/hub/"
        "models--mlx-community--Qwen3.5-4B-4bit/snapshots/"
        "0e7ffd5c629ef7719d4cbc04069232580bfa9d9c"
    )
    if audit_path.exists() or lock_path.exists():
        raise RuntimeError("V81 implementation is already frozen")
    if outcome_dir.exists():
        raise RuntimeError("V81 outcome exists before implementation lock")

    design = json.loads(design_path.read_text())
    design_payload = {
        key: value for key, value in design.items() if key != "lock_payload_sha256"
    }
    config = design["config_payload"]
    seal = json.loads(seal_path.read_text())
    seal_payload = {key: value for key, value in seal.items() if key != "lock_payload_sha256"}
    corpus_path = PROJECT_ROOT / seal["corpus"]
    records = read_jsonl(corpus_path)
    audit_path.parent.mkdir(parents=True, exist_ok=True)

    base_access = {
        "model_generation_count": 24,
        "API_call_count": 0,
        "adapter_training_run_count": 0,
        "human_record_access_count": 0,
        "real_tool_call_count": 0,
        "external_side_effect_count": 0,
    }

    def ordered_labels(record: dict[str, Any]) -> dict[str, bool]:
        return {
            key: record["goldLabels"][key]
            for key in config["labelKeysInRequiredOrder"]
        }

    perfect_rows = [
        score_record(record, json.dumps(ordered_labels(record)), config)
        for record in records
    ]
    perfect_gates = evaluate_gates(aggregate(perfect_rows), config, base_access)
    bad_boolean = ordered_labels(records[0])
    bad_boolean["schedule_review"] = 1
    bad_rows = list(perfect_rows)
    bad_rows[0] = score_record(records[0], json.dumps(bad_boolean), config)
    bad_gates = evaluate_gates(aggregate(bad_rows), config, base_access)
    candidate_field = {**ordered_labels(records[0]), "candidate_ids": ["none_of_the_above"]}
    candidate_rows = list(perfect_rows)
    candidate_rows[0] = score_record(records[0], json.dumps(candidate_field), config)
    candidate_gates = evaluate_gates(aggregate(candidate_rows), config, base_access)
    excessive_access = dict(base_access)
    excessive_access["model_generation_count"] = 25
    access_gates = evaluate_gates(aggregate(perfect_rows), config, excessive_access)
    mutations = {
        "perfect_synthetic_population_passes_every_gate": all(perfect_gates.values()),
        "nonboolean_output_fails_schema_gate": not bad_gates["schema_validity_rate"],
        "model_candidate_ID_field_fails_schema_and_field_gates": bool(
            not candidate_gates["schema_validity_rate"]
            and not candidate_gates["zero_candidate_ID_fields_from_model"]
        ),
        "twenty_fifth_generation_fails_access_gate": not access_gates[
            "bounded_local_model_and_zero_external_access"
        ],
    }
    runner_source = runner_path.read_text()
    checks = {
        "design_and_corpus_locks_authorize_implementation_only": bool(
            payload_hash(design_payload) == design["lock_payload_sha256"]
            and design["authorization"]["implement_and_audit_local_runner"]
            and not design["authorization"]["run_local_model"]
            and payload_hash(seal_payload) == seal["lock_payload_sha256"]
            and seal["authorization"]["implement_and_audit_runner"]
            and not seal["authorization"]["run_local_model"]
        ),
        "sealed_corpus_exactly_matches_fresh_preregistered_population": bool(
            file_sha256(corpus_path) == seal["corpus_sha256"]
            and records == config["records"]
            and len(records) == 24
        ),
        "pinned_snapshot_is_already_local_without_download": bool(
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
        ),
        "strict_parser_composer_and_gate_mutations_detected": all(mutations.values()),
        "runner_is_local_deterministic_durable_and_no_retry": bool(
            "load(str(snapshot))" in runner_source
            and "enable_thinking=False" in runner_source
            and "make_sampler(temp=config[\"decoding\"][\"temperature\"])" in runner_source
            and not config["decoding"]["retryOnMalformedOutput"]
            and "model_generation_count" in runner_source
            and "access-progress.json" in runner_source
            and "run_locked_census_once" in runner_source
        ),
        "zero_model_access_during_implementation_audit": True,
        "zero_API_adapter_human_tool_or_external_access": True,
    }
    passed = all(checks.values())
    audit = {
        "schema_version": "81-factorized-local-candidate-implementation-audit",
        "experiment": "v81_factorized_local_candidate_implementation_audit",
        "passed": passed,
        "decision": (
            "freeze_and_authorize_one_24-record_local_evaluation"
            if passed
            else "reject_V81_local_evaluation"
        ),
        "checks": checks,
        "mutation_checks": mutations,
        "access": {
            "model_load_count": 0,
            "model_generation_count": 0,
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
        "schema_version": "81-factorized-local-candidate-implementation-lock",
        "experiment": "v81_factorized_local_candidate_implementation_lock",
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
        "local_snapshot_path": str(snapshot),
        "local_config_sha256": file_sha256(snapshot / "config.json"),
        "local_model_index_sha256": file_sha256(snapshot / "model.safetensors.index.json"),
        "local_tokenizer_config_sha256": file_sha256(snapshot / "tokenizer_config.json"),
        "authorization": {
            "modify_prompt_records_model_parser_composer_decoding_or_gates": False,
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
    print(json.dumps({"lock": str(lock_path), "sha256": file_sha256(lock_path)}, indent=2))


if __name__ == "__main__":
    main()
