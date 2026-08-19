#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from audit_and_freeze_v167_exact_evidence_gathering_planner import payload_hash
from retrieve_v218_mondo_artifacts import dependency_hashes_exact
from run_v218_mondo_artifact_population import read_jsonl
from v10_protocol import file_sha256
from v218_mondo_artifact_population import audit_population, score_population
from v22r2_grounding import PROJECT_ROOT


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def main() -> None:
    lock_path = PROJECT_ROOT / "configs/v218-mondo-artifact-population-lock.json"
    lock = json.loads(lock_path.read_text())
    audit_path = PROJECT_ROOT / "outputs/v218-mondo-artifact-population/outcome-audit.json"
    outcome_path = PROJECT_ROOT / "configs/v218-mondo-artifact-population-outcome-lock.json"
    results_path = PROJECT_ROOT / "docs/v218-mondo-artifact-population-results.md"
    if audit_path.exists() or outcome_path.exists():
        raise RuntimeError("V218 outcome is already audited or frozen")
    config = lock["config_payload"]
    artifacts = {key: PROJECT_ROOT / value for key, value in config["artifacts"].items()}
    retrieval = json.loads(artifacts["retrievalManifest"].read_text())
    parser_control = json.loads(artifacts["parserControl"].read_text())
    public_records = read_jsonl(artifacts["developmentPublic"]) + read_jsonl(artifacts["protectedPublic"])
    truth_records = read_jsonl(artifacts["developmentTruth"]) + read_jsonl(artifacts["protectedTruth"])
    split = json.loads(artifacts["split"].read_text())
    population_manifest = json.loads(artifacts["populationManifest"].read_text())
    summary = json.loads(artifacts["summary"].read_text())
    result = json.loads(artifacts["result"].read_text())
    metrics = score_population(retrieval, parser_control, public_records, truth_records, split, population_manifest, config, PROJECT_ROOT)
    rebuilt = audit_population(metrics, summary["access"], config)
    raw_checks = []
    for row in retrieval["attempts"]:
        if row["success"]:
            path = PROJECT_ROOT / row["raw_path"]
            raw_checks.append(path.is_file() and file_sha256(path) == row["sha256"] and path.stat().st_size == row["byte_count"])
    hash_fields = {
        "retrieval_manifest_sha256": "retrievalManifest",
        "parser_control_sha256": "parserControl",
        "population_manifest_sha256": "populationManifest",
        "development_public_sha256": "developmentPublic",
        "development_truth_sha256": "developmentTruth",
        "protected_public_sha256": "protectedPublic",
        "protected_truth_sha256": "protectedTruth",
        "split_sha256": "split",
    }
    checks = {
        "design_lock_and_dependencies_exact": dependency_hashes_exact(lock),
        "raw_payloads_and_all_derived_artifact_hashes_are_frozen": bool(
            all(raw_checks)
            and all(summary[field] == file_sha256(artifacts[key]) for field, key in hash_fields.items())
        ),
        "summary_metrics_access_and_scientific_audit_reconstruct_exactly": bool(
            summary["metrics"] == metrics and summary["audit"] == rebuilt and summary["claim_boundary"] == config["claimBoundary"]
        ),
        "result_reconstructs_for_positive_or_negative_scientific_branch": bool(
            result["passed"] == rebuilt["passed"]
            and result["branch"] == rebuilt["branch"]
            and result["decision"] == rebuilt["decision"]
        ),
        "results_document_exists": results_path.is_file(),
    }
    passed = all(checks.values())
    audit = {
        "schema_version": "218-mondo-artifact-population-outcome-audit",
        "experiment": lock["experiment"],
        "passed": passed,
        "scientific_passed": rebuilt["passed"],
        "branch": rebuilt["branch"],
        "decision": "freeze_verified_V218" if passed else "freeze_failed_V218_verification",
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
        "retrieval_manifest": artifacts["retrievalManifest"],
        "parser_control": artifacts["parserControl"],
        "population_manifest": artifacts["populationManifest"],
        "development_public": artifacts["developmentPublic"],
        "development_truth": artifacts["developmentTruth"],
        "protected_public": artifacts["protectedPublic"],
        "protected_truth": artifacts["protectedTruth"],
        "split": artifacts["split"],
        "summary": artifacts["summary"],
        "result": artifacts["result"],
        "results_document": results_path,
        "verifier": PROJECT_ROOT / lock["verifier"],
    }
    outcome: dict[str, Any] = {
        "schema_version": "218-mondo-artifact-population-outcome-lock",
        "experiment": lock["experiment"],
        "outcome": {
            "verification_passed": True,
            "scientific_passed": rebuilt["passed"],
            "branch": rebuilt["branch"],
            "decision": rebuilt["decision"],
            "metrics": metrics,
        },
        "authorization": {
            "design_V219_deterministic_controls": rebuilt["passed"],
            "open_protected_or_run_model": False,
            "register_mutate_service_act_execute": False,
        },
    }
    for key, path in dependencies.items():
        outcome[key] = str(path.relative_to(PROJECT_ROOT))
        outcome[f"{key}_sha256"] = file_sha256(path)
    outcome["raw_payload_hashes"] = {row["payload_id"]: row["sha256"] for row in retrieval["attempts"] if row["success"]}
    outcome["lock_payload_sha256"] = payload_hash(outcome)
    write_json(outcome_path, outcome)
    print(json.dumps(audit, indent=2, sort_keys=True))
    print(json.dumps({"outcome_lock": str(outcome_path.relative_to(PROJECT_ROOT)), "sha256": file_sha256(outcome_path)}, indent=2))


if __name__ == "__main__":
    main()
