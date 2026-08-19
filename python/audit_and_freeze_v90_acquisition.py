#!/usr/bin/env python3
"""Freeze V90 corpus, protocol, downloader, and runner before model acquisition."""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Any

from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT
from v90_capacity_generation_protocol import aggregate, evaluate_condition_gates, quality_gate_pass, score_response


def payload_hash(value: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def main() -> None:
    design_path = PROJECT_ROOT / "configs/v90-capacity-generation-design-lock.json"
    seal_path = PROJECT_ROOT / "data/v90-capacity-generation/corpus-seal.json"
    protocol_path = PROJECT_ROOT / "python/v90_capacity_generation_protocol.py"
    tests_path = PROJECT_ROOT / "python/test_v90_capacity_generation_protocol.py"
    downloader_path = PROJECT_ROOT / "python/download_v90_models.py"
    runner_path = PROJECT_ROOT / "python/run_v90_capacity_condition_mlx.py"
    auditor_path = PROJECT_ROOT / "python/audit_and_freeze_v90_acquisition.py"
    implementation_auditor_path = PROJECT_ROOT / "python/audit_and_freeze_v90_implementation.py"
    harness_path = PROJECT_ROOT / "python/locked_census_harness.py"
    audit_path = PROJECT_ROOT / "outputs/v90-capacity-generation/acquisition-audit.json"
    lock_path = PROJECT_ROOT / "configs/v90-capacity-generation-acquisition-lock.json"
    acquisition_root = PROJECT_ROOT / "outputs/v90-capacity-generation/model-acquisition"
    evaluation_root = PROJECT_ROOT / "outputs/v90-capacity-generation/evaluation"
    if audit_path.exists() or lock_path.exists() or acquisition_root.exists() or evaluation_root.exists():
        raise RuntimeError("V90 acquisition is already frozen, materialized, or evaluated")

    design = json.loads(design_path.read_text())
    design_payload = {key: value for key, value in design.items() if key != "lock_payload_sha256"}
    seal = json.loads(seal_path.read_text())
    seal_payload = {key: value for key, value in seal.items() if key != "lock_payload_sha256"}
    corpus_path = PROJECT_ROOT / seal["corpus"]
    records = read_jsonl(corpus_path)
    records_by_id = {record["id"]: record for record in records}
    config = design["config_payload"]
    perfect_rows = []
    for record in records:
        row = score_response(record, json.dumps({
            "intent_candidates": record["gold"]["intent_candidates"],
            "state_slot_key_candidates": record["gold"]["state_slot_key_candidates"],
        }))
        row["name"] = record["id"]
        perfect_rows.append(row)
    full_access = {
        "model_load_count": 1,
        "model_generation_count": 48,
        "LLM_API_call_count": 0,
        "adapter_training_run_count": 0,
        "manual_utterance_inspection_count": 0,
        "real_service_call_count": 0,
        "external_side_effect_count": 0,
    }
    oracle_metrics = aggregate(perfect_rows, records_by_id)
    oracle_gates = evaluate_condition_gates(oracle_metrics, config, full_access)
    malformed = score_response(records[0], "not-json")
    action_key = score_response(records[0], json.dumps({
        "intent_candidates": records[0]["gold"]["intent_candidates"],
        "state_slot_key_candidates": records[0]["gold"]["state_slot_key_candidates"],
        "action": "execute",
    }))
    strata = Counter((record["service"], record["gold"]["active_intent"]) for record in records)
    runner_source = runner_path.read_text()
    checks = {
        "design_and_corpus_locks_are_exact_and_authorize_implementation_only": bool(
            payload_hash(design_payload) == design["lock_payload_sha256"]
            and design["authorization"]["implement_and_audit_acquisition_and_runner_after_corpus_seal"]
            and not design["authorization"]["run_local_model_before_implementation_lock"]
            and payload_hash(seal_payload) == seal["lock_payload_sha256"]
            and seal["authorization"]["implement_and_audit_model_acquisition_and_runner"]
            and not seal["authorization"]["download_pinned_model_weights_after_acquisition_code_lock"]
        ),
        "sealed_population_hash_cardinality_strata_and_dialogue_uniqueness_are_exact": bool(
            file_sha256(corpus_path) == seal["corpus_sha256"]
            and len(records) == len(records_by_id) == seal["record_count"] == 48
            and seal["dialogue_count"] == len({record["source_dialogue_id"] for record in records}) == 48
            and {f"{k[0]}::{k[1]}": v for k, v in sorted(strata.items())} == seal["stratum_counts"]
            and all(not record["provenance"]["deployable"] and not record["provenance"]["executable"] for record in records)
        ),
        "oracle_population_passes_every_quality_and_access_gate": bool(
            all(oracle_gates.values()) and quality_gate_pass(oracle_gates)
        ),
        "parser_fails_closed_on_malformed_and_action_bearing_outputs": bool(
            not malformed["ontology_conformant"]
            and not action_key["ontology_conformant"]
            and all(not row["executable"] and row["permanently_non_deployable"] for row in perfect_rows)
        ),
        "runner_is_condition_scoped_local_deterministic_no_thinking_no_retry_and_durable": bool(
            "--condition" in runner_source
            and "load(str(snapshot))" in runner_source
            and "enable_thinking=False" in runner_source
            and "make_sampler(temp=config[\"decoding\"][\"temperature\"])" in runner_source
            and "run_locked_census_once" in runner_source
            and "access-progress.json" in runner_source
            and not config["decoding"]["retryOnMalformedOutput"]
        ),
        "download_is_separate_from_model_load_and_supports_transport_resume_only": bool(
            "snapshot_download" in downloader_path.read_text()
            and "model_load_count\": 0" in downloader_path.read_text()
            and "model_generation_count\": 0" in downloader_path.read_text()
        ),
        "all_locked_code_and_test_dependencies_exist": all(path.is_file() for path in (
            protocol_path, tests_path, downloader_path, runner_path, auditor_path,
            implementation_auditor_path, harness_path,
        )),
        "zero_preacquisition_weight_model_API_training_manual_service_or_side_effect_access": bool(
            seal["model_weight_download_count"] == 0
            and seal["model_load_count"] == 0
            and seal["model_generation_count"] == 0
            and seal["manual_utterance_inspection_count"] == 0
        ),
    }
    passed = all(checks.values())
    audit = {
        "schema_version": "90-capacity-generation-acquisition-audit",
        "experiment": "v90_capacity_generation_acquisition_audit",
        "passed": passed,
        "decision": "freeze_and_authorize_pinned_resumable_model_file_acquisition_only" if passed else "reject_V90_model_acquisition",
        "checks": checks,
        "oracle_metrics": oracle_metrics,
        "oracle_gates": oracle_gates,
        "access": {
            "model_weight_download_count": 0,
            "model_load_count": 0,
            "model_generation_count": 0,
            "LLM_API_call_count": 0,
            "adapter_training_run_count": 0,
            "manual_utterance_inspection_count": 0,
            "real_service_call_count": 0,
            "external_side_effect_count": 0,
        },
    }
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    if not passed:
        print(json.dumps(audit, indent=2, sort_keys=True))
        raise SystemExit(1)

    lock = {
        "schema_version": "90-capacity-generation-acquisition-lock",
        "experiment": "v90_capacity_generation_acquisition_lock",
        "design_lock": str(design_path.relative_to(PROJECT_ROOT)),
        "design_lock_sha256": file_sha256(design_path),
        "config_payload": config,
        "corpus_seal": str(seal_path.relative_to(PROJECT_ROOT)),
        "corpus_seal_sha256": file_sha256(seal_path),
        "corpus": str(corpus_path.relative_to(PROJECT_ROOT)),
        "corpus_sha256": file_sha256(corpus_path),
        "protocol": str(protocol_path.relative_to(PROJECT_ROOT)),
        "protocol_sha256": file_sha256(protocol_path),
        "tests": str(tests_path.relative_to(PROJECT_ROOT)),
        "tests_sha256": file_sha256(tests_path),
        "downloader": str(downloader_path.relative_to(PROJECT_ROOT)),
        "downloader_sha256": file_sha256(downloader_path),
        "runner": str(runner_path.relative_to(PROJECT_ROOT)),
        "runner_sha256": file_sha256(runner_path),
        "census_harness": str(harness_path.relative_to(PROJECT_ROOT)),
        "census_harness_sha256": file_sha256(harness_path),
        "acquisition_auditor": str(auditor_path.relative_to(PROJECT_ROOT)),
        "acquisition_auditor_sha256": file_sha256(auditor_path),
        "implementation_auditor": str(implementation_auditor_path.relative_to(PROJECT_ROOT)),
        "implementation_auditor_sha256": file_sha256(implementation_auditor_path),
        "acquisition_audit": str(audit_path.relative_to(PROJECT_ROOT)),
        "acquisition_audit_sha256": file_sha256(audit_path),
        "authorization": {
            "modify_corpus_models_prompt_protocol_runner_decoding_gates_or_decisions": False,
            "download_pinned_snapshots_with_resumable_transport": True,
            "maximum_snapshot_manifest_count": 4,
            "load_or_generate_from_any_model": False,
            "manually_inspect_source_language": False,
            "run_API_model_or_train_adapter": False,
            "grant_model_belief_action_or_execution_authority": False,
            "perform_real_service_call_or_external_side_effect": False,
        },
    }
    lock["lock_payload_sha256"] = payload_hash(lock)
    lock_path.write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n")
    print(json.dumps(audit, indent=2, sort_keys=True))
    print(json.dumps({"lock": str(lock_path.relative_to(PROJECT_ROOT)), "sha256": file_sha256(lock_path)}, indent=2))


if __name__ == "__main__":
    main()
