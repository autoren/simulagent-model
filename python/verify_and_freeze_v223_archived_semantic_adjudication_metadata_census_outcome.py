#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from audit_and_freeze_v167_exact_evidence_gathering_planner import payload_hash
from run_v223_archived_semantic_adjudication_metadata_census import access_ledger, dependency_hashes_exact
from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT
from v223_archived_semantic_adjudication_metadata_census import audit_census, score_census


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def main() -> None:
    lock_path = PROJECT_ROOT / "configs/v223-archived-semantic-adjudication-metadata-census-lock.json"
    lock = json.loads(lock_path.read_text())
    audit_path = PROJECT_ROOT / "outputs/v223-archived-semantic-adjudication-metadata-census/outcome-audit.json"
    outcome_path = PROJECT_ROOT / "configs/v223-archived-semantic-adjudication-metadata-census-outcome-lock.json"
    results_path = PROJECT_ROOT / "docs/v223-archived-semantic-adjudication-metadata-census-results.md"
    if audit_path.exists() or outcome_path.exists():
        raise RuntimeError("V223 outcome is already audited or frozen")
    config = lock["config_payload"]
    artifacts = {key: PROJECT_ROOT / value for key, value in config["artifacts"].items()}
    manifest = json.loads(artifacts["retrievalManifest"].read_text())
    evidence = json.loads(artifacts["evidence"].read_text())
    summary = json.loads(artifacts["summary"].read_text())
    result = json.loads(artifacts["result"].read_text())
    metrics = score_census(manifest, evidence, config, PROJECT_ROOT)
    rebuilt = audit_census(metrics, access_ledger(), config)
    snapshot_checks = []
    for row in manifest["attempts"]:
        if row["success"]:
            path = PROJECT_ROOT / row["snapshot_path"]
            snapshot_checks.append(
                path.is_file()
                and file_sha256(path) == row["sha256"]
                and path.stat().st_size == row["byte_count"]
            )
    checks = {
        "design_lock_and_dependencies_are_exact": dependency_hashes_exact(lock),
        "manifest_evidence_and_metadata_snapshots_are_exact": bool(
            summary["retrieval_manifest_sha256"] == file_sha256(artifacts["retrievalManifest"])
            and summary["evidence_sha256"] == file_sha256(artifacts["evidence"])
            and all(snapshot_checks)
        ),
        "summary_metrics_access_and_audit_reconstruct_exactly": bool(
            summary["metrics"] == metrics
            and summary["access"] == access_ledger()
            and summary["audit"] == rebuilt
            and rebuilt["passed"]
        ),
        "result_reconstructs_for_positive_or_negative_eligibility_branch": bool(
            result["passed"]
            and result["branch"] == rebuilt["branch"]
            and result["decision"] == rebuilt["decision"]
        ),
        "results_document_exists": results_path.is_file(),
    }
    passed = all(checks.values())
    audit = {
        "schema_version": "223-archived-semantic-adjudication-metadata-census-outcome-audit",
        "experiment": config["experiment"],
        "passed": passed,
        "branch": rebuilt["branch"],
        "decision": "freeze_verified_V223" if passed else "freeze_failed_V223_verification",
        "checks": checks,
        "formal_task_record_body_read_count": 0,
        "metrics": metrics,
    }
    write_json(audit_path, audit)
    if not passed:
        print(json.dumps(audit, indent=2, sort_keys=True))
        raise SystemExit(1)
    eligible = rebuilt["branch"] == "SOURCE_SPECIFIC_ACQUISITION_DESIGN_ELIGIBLE"
    dependencies = {
        "design_lock": lock_path,
        "audit": audit_path,
        "retrieval_manifest": artifacts["retrievalManifest"],
        "evidence": artifacts["evidence"],
        "summary": artifacts["summary"],
        "result": artifacts["result"],
        "results_document": results_path,
        "verifier": PROJECT_ROOT / lock["verifier"],
    }
    outcome: dict[str, Any] = {
        "schema_version": "223-archived-semantic-adjudication-metadata-census-outcome-lock",
        "experiment": config["experiment"],
        "outcome": {
            "passed": True,
            "branch": rebuilt["branch"],
            "decision": rebuilt["decision"],
            "selected_source_specific_candidate_ids": result["selected_source_specific_candidate_ids"],
            "formal_task_record_body_read_count": 0,
            "metrics": metrics,
        },
        "authorization": {
            "design_source_specific_metadata_first_acquisition_and_identifiability_stage": eligible,
            "open_task_record_language_or_run_model": False,
            "register_mutate_service_act_execute": False,
        },
    }
    for key, path in dependencies.items():
        outcome[key] = str(path.relative_to(PROJECT_ROOT))
        outcome[f"{key}_sha256"] = file_sha256(path)
    outcome["snapshot_hashes"] = {
        row["snapshot_path"]: row["sha256"] for row in manifest["attempts"] if row["success"]
    }
    outcome["lock_payload_sha256"] = payload_hash(outcome)
    write_json(outcome_path, outcome)
    print(json.dumps(audit, indent=2, sort_keys=True))
    print(json.dumps({"outcome_lock": str(outcome_path.relative_to(PROJECT_ROOT)), "sha256": file_sha256(outcome_path)}, indent=2))


if __name__ == "__main__":
    main()
