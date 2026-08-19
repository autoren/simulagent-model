#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from audit_and_freeze_v165_factored_ontology_identifiability_population import valid_lock
from v10_protocol import file_sha256
from v215_external_semantic_resource_metadata_census import audit_census, score_census
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
    lock_path = PROJECT_ROOT / "configs/v215-external-semantic-resource-metadata-census-lock.json"
    lock = json.loads(lock_path.read_text())
    if not dependency_hashes_exact(lock):
        raise RuntimeError("V215 design lock or dependency hash mismatch")
    if not lock["authorization"]["score_one_frozen_metadata_census"]:
        raise RuntimeError("V215 census scoring is not authorized")
    config = lock["config_payload"]
    artifacts = {key: PROJECT_ROOT / value for key, value in config["artifacts"].items()}
    for key in ("summary", "result"):
        if artifacts[key].exists():
            raise RuntimeError("V215 formal census output already exists")
    manifest = json.loads(artifacts["retrievalManifest"].read_text())
    evidence = json.loads(artifacts["evidence"].read_text())
    metrics = score_census(manifest, evidence, config, PROJECT_ROOT)
    access = {
        "metadata_census_run_count": 1,
        "bulk_ontology_payload_download_count": 0,
        "alignment_payload_download_count": 0,
        "test_suite_payload_download_count": 0,
        "v213_protected_access_count": 0,
        "model_load_count": 0,
        "model_generation_count": 0,
        "api_call_count": 0,
        "training_run_count": 0,
        "ontology_registration_count": 0,
        "trusted_state_mutation_count": 0,
        "service_action_count": 0,
        "external_side_effect_count": 0,
        "actual_execution_count": 0,
    }
    audit = audit_census(metrics, access, config)
    summary = {
        "schema_version": "215-external-semantic-resource-metadata-census-summary",
        "experiment": config["experiment"],
        "claim_boundary": config["claimBoundary"],
        "retrieval_manifest_sha256": file_sha256(artifacts["retrievalManifest"]),
        "evidence_sha256": file_sha256(artifacts["evidence"]),
        "metrics": metrics,
        "access": access,
        "audit": audit,
    }
    result = {
        "schema_version": "215-external-semantic-resource-metadata-census-result",
        "experiment": config["experiment"],
        "passed": audit["passed"],
        "branch": audit["branch"],
        "decision": audit["decision"],
        "claim_boundary": config["claimBoundary"],
        "selected_payload_candidate_ids": evidence["recommended_payload_candidate_ids"],
        "selected_validation_control_ids": evidence["recommended_validation_control_ids"],
        "authorization": {
            "design_separate_bounded_payload_population_stage": audit["passed"],
            "download_payload_or_run_model_without_separate_lock": False,
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
