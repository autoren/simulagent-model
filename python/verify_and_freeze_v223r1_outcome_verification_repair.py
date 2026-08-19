#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from audit_and_freeze_v165_factored_ontology_identifiability_population import valid_lock
from audit_and_freeze_v167_exact_evidence_gathering_planner import payload_hash
from run_v223_archived_semantic_adjudication_metadata_census import access_ledger
from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT
from v223_archived_semantic_adjudication_metadata_census import audit_census, score_census
from v223r1_outcome_verification_repair import failed_checks, positive_outcome_matches


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def dependency_hashes_exact(lock: dict[str, Any]) -> bool:
    keys = [key for key in lock if not key.endswith("_sha256") and f"{key}_sha256" in lock]
    return valid_lock(lock) and all(
        file_sha256(PROJECT_ROOT / lock[key]) == lock[f"{key}_sha256"] for key in keys
    )


def main() -> None:
    lock_path = PROJECT_ROOT / "configs/v223r1-outcome-verification-repair-lock.json"
    lock = json.loads(lock_path.read_text())
    output_root = PROJECT_ROOT / "outputs/v223r1-outcome-verification-repair"
    audit_path = output_root / "outcome-audit.json"
    outcome_path = PROJECT_ROOT / "configs/v223r1-outcome-verification-repair-outcome-lock.json"
    if audit_path.exists() or outcome_path.exists():
        raise RuntimeError("V223r1 outcome is already audited or frozen")
    if not dependency_hashes_exact(lock):
        raise RuntimeError("V223r1 design lock or dependency hash mismatch")
    if not lock["authorization"]["verify_and_freeze_exact_existing_V223_positive_once"]:
        raise RuntimeError("V223r1 verification is not authorized")
    repair = lock["config_payload"]
    invariant = repair["repairInvariant"]
    parent_lock = json.loads((PROJECT_ROOT / lock["parent_V223_design_lock"]).read_text())
    config = parent_lock["config_payload"]
    artifacts = {key: PROJECT_ROOT / value for key, value in config["artifacts"].items()}
    manifest = json.loads(artifacts["retrievalManifest"].read_text())
    evidence = json.loads(artifacts["evidence"].read_text())
    summary = json.loads(artifacts["summary"].read_text())
    result = json.loads(artifacts["result"].read_text())
    original_failed_audit = json.loads((PROJECT_ROOT / lock["parent_V223_failed_outcome_audit"]).read_text())
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
        "repair_lock_and_every_dependency_hash_are_exact": dependency_hashes_exact(lock),
        "original_failure_is_preserved_and_exactly_non_scientific": bool(
            original_failed_audit.get("passed") is False
            and failed_checks(original_failed_audit) == sorted(invariant["expectedOriginalFailedChecks"])
            and original_failed_audit.get("branch") == invariant["expectedV223Branch"]
        ),
        "metadata_snapshots_and_firewall_remain_exact": bool(
            all(snapshot_checks)
            and metrics["formal_task_record_body_read_count"] == 0
            and metrics["issue_proposal_comment_pull_or_archive_record_request_count"] == 0
        ),
        "metrics_scientific_audit_summary_and_result_reconstruct_exactly": bool(
            metrics == summary["metrics"]
            and rebuilt == summary["audit"]
            and rebuilt["passed"]
            and positive_outcome_matches(summary, result, invariant)
        ),
        "repair_authority_remains_verification_only": bool(
            not lock["authorization"]["retrieve_reassess_change_gate_open_language_or_run_model"]
            and not lock["authorization"]["register_mutate_service_act_execute"]
        ),
    }
    passed = all(checks.values())
    audit = {
        "schema_version": "223r1-outcome-verification-repair-outcome-audit",
        "experiment": repair["experiment"],
        "passed": passed,
        "V223_scientific_passed": rebuilt["passed"],
        "V223_branch": rebuilt["branch"],
        "decision": repair["decisionRule"]["ifRepairIntegrityPasses"] if passed else repair["decisionRule"]["otherwise"],
        "checks": checks,
        "metrics": metrics,
        "formal_task_record_body_read_count": 0,
    }
    write_json(audit_path, audit)
    if not passed:
        print(json.dumps(audit, indent=2, sort_keys=True))
        raise SystemExit(1)
    dependencies: dict[str, Path] = {
        "repair_design_lock": lock_path,
        "repair_audit": audit_path,
    }
    for key in [key for key in lock if not key.endswith("_sha256") and f"{key}_sha256" in lock]:
        dependencies[f"frozen_{key}"] = PROJECT_ROOT / lock[key]
    outcome: dict[str, Any] = {
        "schema_version": "223r1-outcome-verification-repair-outcome-lock",
        "experiment": repair["experiment"],
        "outcome": {
            "repair_passed": True,
            "V223_scientific_passed": True,
            "V223_branch": rebuilt["branch"],
            "V223_decision": rebuilt["decision"],
            "selected_source_specific_candidate_ids": result["selected_source_specific_candidate_ids"],
            "original_failed_checks": failed_checks(original_failed_audit),
            "metrics": metrics,
            "formal_task_record_body_read_count": 0,
        },
        "authorization": {
            "design_V224_Mondo_metadata_first_record_disposition_census": True,
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

