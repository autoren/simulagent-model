#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from audit_and_freeze_v165_factored_ontology_identifiability_population import valid_lock
from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT
from v223_archived_semantic_adjudication_metadata_census import audit_census, score_census


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def dependency_hashes_exact(lock: dict[str, Any]) -> bool:
    keys = [key for key in lock if not key.endswith("_sha256") and f"{key}_sha256" in lock]
    return valid_lock(lock) and all(
        file_sha256(PROJECT_ROOT / lock[key]) == lock[f"{key}_sha256"] for key in keys
    )


def access_ledger() -> dict[str, int]:
    return {
        "metadata_census_run_count": 1,
        "formal_task_record_body_read_count": 0,
        "issue_proposal_comment_pull_or_archive_record_request_count": 0,
        "protected_research_record_read_count": 0,
        "model_load_count": 0,
        "model_generation_count": 0,
        "model_api_call_count": 0,
        "training_run_count": 0,
        "ontology_registration_count": 0,
        "trusted_state_mutation_count": 0,
        "service_action_count": 0,
        "external_side_effect_count": 0,
        "actual_execution_count": 0,
    }


def main() -> None:
    lock_path = PROJECT_ROOT / "configs/v223-archived-semantic-adjudication-metadata-census-lock.json"
    lock = json.loads(lock_path.read_text())
    if not dependency_hashes_exact(lock):
        raise RuntimeError("V223 design lock or dependency hash mismatch")
    if not lock["authorization"]["score_one_metadata_census"]:
        raise RuntimeError("V223 census scoring is not authorized")
    config = lock["config_payload"]
    artifacts = {key: PROJECT_ROOT / value for key, value in config["artifacts"].items()}
    for key in ("summary", "result"):
        if artifacts[key].exists():
            raise RuntimeError("V223 formal census output already exists")
    manifest = json.loads(artifacts["retrievalManifest"].read_text())
    evidence = json.loads(artifacts["evidence"].read_text())
    metrics = score_census(manifest, evidence, config, PROJECT_ROOT)
    access = access_ledger()
    audit = audit_census(metrics, access, config)
    if not audit["passed"]:
        raise RuntimeError("V223 census integrity or access audit failed")
    summary = {
        "schema_version": "223-archived-semantic-adjudication-metadata-census-summary",
        "experiment": config["experiment"],
        "claim_boundary": config["claimBoundary"],
        "retrieval_manifest_sha256": file_sha256(artifacts["retrievalManifest"]),
        "evidence_sha256": file_sha256(artifacts["evidence"]),
        "metrics": metrics,
        "access": access,
        "audit": audit,
    }
    eligible = audit["branch"] == "SOURCE_SPECIFIC_ACQUISITION_DESIGN_ELIGIBLE"
    result = {
        "schema_version": "223-archived-semantic-adjudication-metadata-census-result",
        "experiment": config["experiment"],
        "passed": True,
        "branch": audit["branch"],
        "decision": audit["decision"],
        "selected_source_specific_candidate_ids": metrics["recommended_source_specific_candidate_ids"],
        "authorization": {
            "design_source_specific_metadata_first_acquisition_and_identifiability_stage": eligible,
            "open_task_record_language_or_run_model_without_separate_lock": False,
            "register_mutate_service_act_execute": False,
        },
    }
    write_json(artifacts["summary"], summary)
    write_json(artifacts["result"], result)
    print(json.dumps(result, indent=2, sort_keys=True))
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

