#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from run_v223_archived_semantic_adjudication_metadata_census import dependency_hashes_exact
from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT
from v224_mondo_record_disposition_metadata_census import audit_census, score_census


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def access_ledger() -> dict[str, int]:
    return {
        "formal_census_run_count": 1,
        "task_record_title_read_count": 0,
        "task_record_body_read_count": 0,
        "comment_or_review_text_read_count": 0,
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
    lock_path = PROJECT_ROOT / "configs/v224-mondo-record-disposition-metadata-census-lock.json"
    lock = json.loads(lock_path.read_text())
    if not dependency_hashes_exact(lock):
        raise RuntimeError("V224 design lock or dependency hash mismatch")
    config = lock["config_payload"]
    artifacts = {key: PROJECT_ROOT / value for key, value in config["artifacts"].items()}
    for key in ("summary", "result"):
        if artifacts[key].exists():
            raise RuntimeError("V224 formal result already exists")
    required = (
        "scopePolicySnapshot", "queryManifest", "recordMetadata", "preliminaryCensus",
        "releaseMetadata", "releaseIdIndex", "deepAudit",
    )
    if not all(artifacts[key].is_file() for key in required):
        raise RuntimeError("V224 capture artifacts are incomplete")
    query_manifest = json.loads(artifacts["queryManifest"].read_text())
    records = read_jsonl(artifacts["recordMetadata"])
    preliminary = json.loads(artifacts["preliminaryCensus"].read_text())
    deep_rows = read_jsonl(artifacts["deepAudit"])
    metrics = score_census(records, preliminary, deep_rows, query_manifest, config)
    access = access_ledger()
    audit = audit_census(metrics, access, config)
    if not audit["passed"]:
        raise RuntimeError("V224 integrity or access audit failed")
    input_hashes = {key: file_sha256(artifacts[key]) for key in required}
    summary = {
        "schema_version": "224-mondo-record-disposition-metadata-census-summary",
        "experiment": config["experiment"],
        "claim_boundary": config["claimBoundary"],
        "input_hashes": input_hashes,
        "metrics": metrics,
        "access": access,
        "audit": audit,
    }
    eligible = audit["branch"] == "V225_LANGUAGE_DESIGN_ELIGIBLE"
    result = {
        "schema_version": "224-mondo-record-disposition-metadata-census-result",
        "experiment": config["experiment"],
        "passed": True,
        "branch": audit["branch"],
        "decision": audit["decision"],
        "authorization": {
            "design_V225_role_separated_language_acquisition_and_identifiability": eligible,
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

