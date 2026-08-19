#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from run_v223_archived_semantic_adjudication_metadata_census import dependency_hashes_exact
from run_v224_mondo_record_disposition_metadata_census import read_jsonl
from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT
import v224_mondo_record_disposition_metadata_census as core
from v224r2_two_stage_query_execution_repair import install_thin_scorer


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def access_ledger() -> dict[str, int]:
    return {
        "formal_census_run_count": 1,
        "prior_failed_safe_metadata_capture_attempt_count": 2,
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
    lock_path = PROJECT_ROOT / "configs/v224r2-two-stage-query-execution-repair-lock.json"
    lock = json.loads(lock_path.read_text())
    if not dependency_hashes_exact(lock):
        raise RuntimeError("V224r2 repair lock or dependency hash mismatch")
    parent = json.loads((PROJECT_ROOT / lock["parent_V224_design_lock"]).read_text())
    config = parent["config_payload"]
    artifacts = {key: PROJECT_ROOT / value for key, value in config["artifacts"].items()}
    for key in ("summary", "result"):
        if artifacts[key].exists():
            raise RuntimeError("V224r2 formal result already exists")
    required = (
        "scopePolicySnapshot", "queryManifest", "recordMetadata", "preliminaryCensus",
        "releaseMetadata", "releaseIdIndex", "deepAudit",
    )
    if not all(artifacts[key].is_file() for key in required):
        raise RuntimeError("V224r2 capture artifacts are incomplete")
    manifest = json.loads(artifacts["queryManifest"].read_text())
    records = read_jsonl(artifacts["recordMetadata"])
    preliminary = json.loads(artifacts["preliminaryCensus"].read_text())
    deep_rows = read_jsonl(artifacts["deepAudit"])
    install_thin_scorer(core)
    metrics = core.score_census(records, preliminary, deep_rows, manifest, config)
    access = access_ledger()
    audit = core.audit_census(metrics, access, config)
    if not audit["passed"]:
        raise RuntimeError("V224r2 integrity or access audit failed")
    summary = {
        "schema_version": "224r2-two-stage-query-execution-repair-summary",
        "experiment": lock["config_payload"]["experiment"],
        "scientific_experiment": config["experiment"],
        "claim_boundary": config["claimBoundary"],
        "input_hashes": {key: file_sha256(artifacts[key]) for key in required},
        "metrics": metrics,
        "access": access,
        "audit": audit,
    }
    eligible = audit["branch"] == "V225_LANGUAGE_DESIGN_ELIGIBLE"
    result = {
        "schema_version": "224r2-two-stage-query-execution-repair-result",
        "experiment": lock["config_payload"]["experiment"],
        "scientific_experiment": config["experiment"],
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

