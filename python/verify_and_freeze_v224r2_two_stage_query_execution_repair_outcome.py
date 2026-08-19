#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from audit_and_freeze_v167_exact_evidence_gathering_planner import payload_hash
from run_v223_archived_semantic_adjudication_metadata_census import dependency_hashes_exact
from run_v224_mondo_record_disposition_metadata_census import read_jsonl
from run_v224r2_two_stage_query_execution_repair import access_ledger
from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT
import v224_mondo_record_disposition_metadata_census as core
from v224r2_two_stage_query_execution_repair import install_thin_scorer


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def main() -> None:
    lock_path = PROJECT_ROOT / "configs/v224r2-two-stage-query-execution-repair-lock.json"
    lock = json.loads(lock_path.read_text())
    audit_path = PROJECT_ROOT / "outputs/v224r2-two-stage-query-execution-repair/outcome-audit.json"
    outcome_path = PROJECT_ROOT / "configs/v224r2-two-stage-query-execution-repair-outcome-lock.json"
    results_path = PROJECT_ROOT / "docs/v224-mondo-record-disposition-metadata-census-results.md"
    if audit_path.exists() or outcome_path.exists():
        raise RuntimeError("V224r2 outcome already exists")
    parent = json.loads((PROJECT_ROOT / lock["parent_V224_design_lock"]).read_text())
    config = parent["config_payload"]
    artifacts = {key: PROJECT_ROOT / value for key, value in config["artifacts"].items()}
    manifest = json.loads(artifacts["queryManifest"].read_text())
    records = read_jsonl(artifacts["recordMetadata"])
    preliminary = json.loads(artifacts["preliminaryCensus"].read_text())
    deep_rows = read_jsonl(artifacts["deepAudit"])
    summary = json.loads(artifacts["summary"].read_text())
    result = json.loads(artifacts["result"].read_text())
    install_thin_scorer(core)
    metrics = core.score_census(records, preliminary, deep_rows, manifest, config)
    rebuilt = core.audit_census(metrics, access_ledger(), config)
    required = (
        "scopePolicySnapshot", "queryManifest", "recordMetadata", "preliminaryCensus",
        "releaseMetadata", "releaseIdIndex", "deepAudit",
    )
    checks = {
        "repair_lock_and_dependencies_are_exact": dependency_hashes_exact(lock),
        "capture_artifacts_are_exact": all(
            summary["input_hashes"][key] == file_sha256(artifacts[key]) for key in required
        ),
        "metrics_access_audit_and_result_reconstruct_exactly": bool(
            metrics == summary["metrics"] and summary["access"] == access_ledger()
            and rebuilt == summary["audit"] and rebuilt["passed"]
            and result["passed"] and result["branch"] == rebuilt["branch"]
            and result["decision"] == rebuilt["decision"]
        ),
        "two_failed_attempts_and_zero_language_model_effect_access_are_explicit": bool(
            summary["access"]["prior_failed_safe_metadata_capture_attempt_count"] == 2
            and metrics["task_language_persistence_count"] == 0
            and all(
                value == 0 for key, value in access_ledger().items()
                if key not in {"formal_census_run_count", "prior_failed_safe_metadata_capture_attempt_count"}
            )
        ),
        "results_document_exists": results_path.is_file(),
    }
    passed = all(checks.values())
    audit = {
        "schema_version": "224r2-two-stage-query-execution-repair-outcome-audit",
        "experiment": lock["config_payload"]["experiment"],
        "passed": passed,
        "branch": rebuilt["branch"],
        "decision": "freeze_verified_V224r2" if passed else "freeze_failed_V224r2_verification",
        "checks": checks,
        "metrics": metrics,
        "task_record_title_or_body_read_count": 0,
    }
    write_json(audit_path, audit)
    if not passed:
        print(json.dumps(audit, indent=2, sort_keys=True))
        raise SystemExit(1)
    eligible = rebuilt["branch"] == "V225_LANGUAGE_DESIGN_ELIGIBLE"
    dependencies = {
        "repair_design_lock": lock_path, "audit": audit_path, "summary": artifacts["summary"],
        "result": artifacts["result"], "results_document": results_path,
        "verifier": PROJECT_ROOT / lock["verifier"],
    }
    outcome: dict[str, Any] = {
        "schema_version": "224r2-two-stage-query-execution-repair-outcome-lock",
        "experiment": lock["config_payload"]["experiment"],
        "outcome": {
            "repair_passed": True, "V224_scientific_passed": True,
            "V224_branch": rebuilt["branch"], "V224_decision": rebuilt["decision"],
            "metrics": metrics, "prior_failed_capture_attempt_count": 2,
            "task_record_title_or_body_read_count": 0,
        },
        "authorization": {
            "design_V225_role_separated_language_acquisition_and_identifiability": eligible,
            "open_task_record_language_or_run_model": False,
            "register_mutate_service_act_execute": False,
        },
    }
    for key, path in dependencies.items():
        outcome[key] = str(path.relative_to(PROJECT_ROOT))
        outcome[f"{key}_sha256"] = file_sha256(path)
    outcome["capture_artifact_hashes"] = summary["input_hashes"]
    outcome["lock_payload_sha256"] = payload_hash(outcome)
    write_json(outcome_path, outcome)
    print(json.dumps(audit, indent=2, sort_keys=True))
    print(json.dumps({"outcome_lock": str(outcome_path.relative_to(PROJECT_ROOT)), "sha256": file_sha256(outcome_path)}, indent=2))


if __name__ == "__main__":
    main()

