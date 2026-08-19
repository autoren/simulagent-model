#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT
from v100_massive_source import parse_massive_archive
from v106_open_world_benchmark import (
    build_declared_training_records, character_retrieval_observations, fit_character_retrieval,
)
from run_v112_open_world_full_policy_transfer import payload_hash, read_jsonl
from run_v112r1_full_policy_aggregation import load_fixtures
from v113_known_disagreement_rescue import build_census, extract_rescue_features


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def reconstruct(lock: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    records = read_jsonl(PROJECT_ROOT / lock["fresh_language"])
    archive_bytes = (PROJECT_ROOT / lock["source_archive"]).read_bytes()
    source_records, member = parse_massive_archive(
        archive_bytes, lock["V112_config_payload"]["extraction"]["expectedLocaleMemberSuffix"],
    )
    catalog = json.loads((PROJECT_ROOT / lock["visible_catalog"]).read_text())
    training = build_declared_training_records(source_records, catalog)
    retrieval_spec = lock["baseline_config_payload"]["deterministicBaselines"]["character_ngram_retrieval"]
    fitted = fit_character_retrieval(training, retrieval_spec["vectorizer"])
    retrieval = character_retrieval_observations(fitted, records)
    fixtures = load_fixtures(lock)
    observed = {row["record_id"]: fixtures[row["record_id"]] for row in records}
    direct = {identifier: row["parsed_response"] for identifier, row in observed.items()}
    controls = [row for row in fixtures.values() if row["kind"] == "controlled_missing_observation"]
    validity = sum(row["response_valid"] for row in fixtures.values()) / len(fixtures)
    controlled_accuracy = sum(
        row["response_valid"] and row["parsed_response"]["status"] == "ABSTAIN"
        for row in controls
    ) / len(controls)
    features = extract_rescue_features(fitted, records, direct)
    census = build_census(
        records, features, direct, retrieval, lock["V112_config_payload"],
        lock["config_payload"], lock["baseline_config_payload"], validity, controlled_accuracy,
    )
    metadata = {
        "historical_record_count": len(records), "preserved_fixture_count": len(fixtures),
        "source_locale_member": member, "interface_validity": validity,
        "controlled_missing_observation_abstention_accuracy": controlled_accuracy,
    }
    return census, metadata


def main() -> None:
    lock_path = PROJECT_ROOT / "configs/v113-known-disagreement-rescue-census-lock.json"
    output_root = PROJECT_ROOT / "outputs/v113-known-disagreement-rescue-census/historical-census"
    if output_root.exists():
        raise RuntimeError("V113 census may run only once")
    lock = json.loads(lock_path.read_text())
    if payload_hash({key: value for key, value in lock.items() if key != "lock_payload_sha256"}) != lock["lock_payload_sha256"]:
        raise RuntimeError("V113 lock mismatch")
    dependency_keys = (
        "config", "parent_outcome", "V112r1_lock", "V112r1_result", "fresh_language",
        "fixture_manifest", "source_archive", "visible_catalog", "plan", "protocol",
        "tests", "runner", "verifier", "auditor", "design_audit",
    )
    for key in dependency_keys:
        if file_sha256(PROJECT_ROOT / lock[key]) != lock[f"{key}_sha256"]:
            raise RuntimeError(f"V113 dependency drifted: {key}")
    census, metadata = reconstruct(lock)
    feasible = census["feasible_candidate_count"] > 0
    result = {
        "schema_version": "113-known-disagreement-rescue-census-result",
        "experiment": lock["config_payload"]["experiment"], "passed": True,
        "feasible_rescue_exists": feasible,
        "decision": (
            "freeze_selected_rescue_for_separately_locked_new_population_transfer"
            if feasible else "close_simple_disagreement_rescue_family_require_new_evidence_or_policy_structure"
        ),
        "analysis": census, "metadata": metadata,
        "access": {
            "preserved_fixture_automatic_read_count": 240,
            "historical_language_automatic_read_count": 1,
            "protected_test_language_read_count": 0,
            "manual_language_or_raw_response_inspection_count": 0,
            "model_load_count": 0, "model_generation_count": 0, "LLM_API_call_count": 0,
            "adapter_training_run_count": 0, "real_service_call_count": 0,
            "external_side_effect_count": 0,
        },
        "claim_boundary": "historical V112 policy-design census only; no new transfer claim, individual evidence emission, protected test, model inference, API, training, action, execution, service call, or side effect",
    }
    result_path = output_root / "result.json"
    write_json(result_path, result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
