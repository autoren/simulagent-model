#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from audit_and_freeze_v165_factored_ontology_identifiability_population import valid_lock
from v10_protocol import file_sha256
from v217a_independent_source_event_metadata_census import audit_census, score_census
from v22r2_grounding import PROJECT_ROOT


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def dependency_hashes_exact(lock: dict[str, Any]) -> bool:
    keys = [key for key in lock if not key.endswith("_sha256") and f"{key}_sha256" in lock]
    return valid_lock(lock) and all(
        file_sha256(PROJECT_ROOT / lock[key]) == lock[f"{key}_sha256"] for key in keys
    )


def main() -> None:
    lock_path = PROJECT_ROOT / "configs/v217a-independent-source-event-metadata-census-lock.json"
    lock = json.loads(lock_path.read_text())
    if not dependency_hashes_exact(lock):
        raise RuntimeError("V217A design lock or dependency hash mismatch")
    if not lock["authorization"]["score_one_frozen_source_event_census"]:
        raise RuntimeError("V217A census scoring is not authorized")
    config = lock["config_payload"]
    artifacts = {key: PROJECT_ROOT / value for key, value in config["artifacts"].items()}
    if artifacts["summary"].exists() or artifacts["result"].exists():
        raise RuntimeError("V217A formal census output already exists")
    manifest = json.loads(artifacts["retrievalManifest"].read_text())
    evidence = json.loads(artifacts["evidence"].read_text())
    metrics = score_census(manifest, evidence, config, PROJECT_ROOT)
    access = {
        "metadata_capture_run_count": 1,
        "metadata_census_run_count": 1,
        "candidate_payload_download_count": 0,
        "v216_protected_access_count": 0,
        "v213_protected_access_count": 0,
        "model_load_count": 0,
        "model_generation_count": 0,
        "model_api_call_count": 0,
        "training_run_count": 0,
        "ontology_registration_count": 0,
        "trusted_state_mutation_count": 0,
        "service_action_count": 0,
        "external_side_effect_count_beyond_read_only_metadata": 0,
        "actual_execution_count": 0,
    }
    audit = audit_census(metrics, access, config)
    summary = {
        "schema_version": "217a-independent-source-event-metadata-census-summary",
        "experiment": config["experiment"],
        "claim_boundary": config["claimBoundary"],
        "retrieval_manifest_sha256": file_sha256(artifacts["retrievalManifest"]),
        "evidence_sha256": file_sha256(artifacts["evidence"]),
        "metrics": metrics,
        "access": access,
        "audit": audit,
    }
    result = {
        "schema_version": "217a-independent-source-event-metadata-census-result",
        "experiment": config["experiment"],
        "passed": audit["passed"],
        "branch": audit["branch"],
        "decision": audit["decision"],
        "claim_boundary": config["claimBoundary"],
        "selected_source_ids": metrics["selected_source_ids"],
        "authorization": {
            "design_one_fresh_source_payload_protocol": audit["passed"],
            "download_payload_open_protected_or_run_model": False,
            "register_mutate_service_act_execute": False,
        },
    }
    write_json(artifacts["summary"], summary)
    write_json(artifacts["result"], result)
    print(json.dumps(result, indent=2, sort_keys=True))
    print(json.dumps(metrics, indent=2, sort_keys=True))
    if not audit["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

