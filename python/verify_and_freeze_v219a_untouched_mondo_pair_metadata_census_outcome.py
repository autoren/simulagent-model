#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from audit_and_freeze_v167_exact_evidence_gathering_planner import payload_hash
from run_v219a_untouched_mondo_pair_metadata_census import dependency_hashes_exact
from v10_protocol import file_sha256
from v219a_untouched_mondo_pair_metadata_census import audit_census, build_census
from v22r2_grounding import PROJECT_ROOT


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def main() -> None:
    lock_path = PROJECT_ROOT / "configs/v219a-untouched-mondo-pair-metadata-census-lock.json"
    lock = json.loads(lock_path.read_text())
    audit_path = PROJECT_ROOT / "outputs/v219a-untouched-mondo-pair-metadata-census/outcome-audit.json"
    outcome_path = PROJECT_ROOT / "configs/v219a-untouched-mondo-pair-metadata-census-outcome-lock.json"
    results_path = PROJECT_ROOT / "docs/v219a-untouched-mondo-pair-metadata-census-results.md"
    if audit_path.exists() or outcome_path.exists():
        raise RuntimeError("V219A outcome is already audited or frozen")
    config = lock["config_payload"]
    artifacts = {key: PROJECT_ROOT / value for key, value in config["artifacts"].items()}
    snapshot_path = PROJECT_ROOT / config["evidenceSource"]["path"]
    releases = json.loads(snapshot_path.read_text())
    rebuilt_evidence, metrics = build_census(
        releases,
        config,
        snapshot_hash_accurate=file_sha256(snapshot_path) == config["evidenceSource"]["sha256"],
    )
    evidence = json.loads(artifacts["evidence"].read_text())
    summary = json.loads(artifacts["summary"].read_text())
    result = json.loads(artifacts["result"].read_text())
    rebuilt_audit = audit_census(metrics, summary["access"], config)
    checks = {
        "design_lock_and_dependencies_exact": dependency_hashes_exact(lock),
        "snapshot_and_evidence_are_exact": bool(
            file_sha256(snapshot_path) == config["evidenceSource"]["sha256"]
            and summary["evidence_snapshot_sha256"] == file_sha256(snapshot_path)
            and summary["evidence_sha256"] == file_sha256(artifacts["evidence"])
            and evidence == rebuilt_evidence
        ),
        "summary_metrics_access_and_audit_reconstruct_exactly": bool(
            summary["metrics"] == metrics
            and summary["audit"] == rebuilt_audit
            and summary["claim_boundary"] == config["claimBoundary"]
        ),
        "result_reconstructs_for_positive_or_negative_branch": bool(
            result["passed"] == rebuilt_audit["passed"]
            and result["branch"] == rebuilt_audit["branch"]
            and result["decision"] == rebuilt_audit["decision"]
            and result["selected_pair_ids"] == metrics["selected_pair_ids"]
        ),
        "results_document_exists": results_path.is_file(),
    }
    passed = all(checks.values())
    audit = {
        "schema_version": "219a-untouched-mondo-pair-metadata-census-outcome-audit",
        "experiment": lock["experiment"],
        "passed": passed,
        "scientific_passed": rebuilt_audit["passed"],
        "branch": rebuilt_audit["branch"],
        "decision": "freeze_verified_V219A" if passed else "freeze_failed_V219A_verification",
        "checks": checks,
        "metrics": metrics,
    }
    write_json(audit_path, audit)
    if not passed:
        print(json.dumps(audit, indent=2, sort_keys=True))
        raise SystemExit(1)
    dependencies = {
        "design_lock": lock_path,
        "audit": audit_path,
        "evidence_snapshot": snapshot_path,
        "evidence": artifacts["evidence"],
        "summary": artifacts["summary"],
        "result": artifacts["result"],
        "results_document": results_path,
        "verifier": PROJECT_ROOT / lock["verifier"],
    }
    selected = evidence["selected_pair_assessments"]
    outcome: dict[str, Any] = {
        "schema_version": "219a-untouched-mondo-pair-metadata-census-outcome-lock",
        "experiment": lock["experiment"],
        "outcome": {
            "verification_passed": True,
            "scientific_passed": rebuilt_audit["passed"],
            "branch": rebuilt_audit["branch"],
            "decision": rebuilt_audit["decision"],
            "selected_pair_ids": metrics["selected_pair_ids"],
            "selected_pair_assessments": selected,
            "metrics": metrics,
        },
        "authorization": {
            "design_one_untouched_pair_payload_protocol": rebuilt_audit["passed"],
            "retrieve_payload_or_evaluate_method": False,
            "open_protected_or_run_model": False,
            "register_mutate_service_act_execute": False,
        },
    }
    for key, path in dependencies.items():
        outcome[key] = str(path.relative_to(PROJECT_ROOT))
        outcome[f"{key}_sha256"] = file_sha256(path)
    outcome["lock_payload_sha256"] = payload_hash(outcome)
    write_json(outcome_path, outcome)
    print(json.dumps(audit, indent=2, sort_keys=True))
    print(json.dumps({"outcome_lock": str(outcome_path.relative_to(PROJECT_ROOT)), "sha256": file_sha256(outcome_path)}, indent=2))


if __name__ == "__main__":
    main()
