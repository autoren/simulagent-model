#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from audit_and_freeze_v167_exact_evidence_gathering_planner import payload_hash
from retrieve_v216_bounded_external_artifacts import dependency_hashes_exact
from run_v216_bounded_external_artifact_population import read_jsonl
from v10_protocol import file_sha256
from v216_bounded_external_artifact_population import audit_population, score_population
from v22r2_grounding import PROJECT_ROOT


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def main() -> None:
    lock_path = PROJECT_ROOT / "configs/v216-bounded-external-artifact-population-lock.json"
    lock = json.loads(lock_path.read_text())
    audit_path = PROJECT_ROOT / "outputs/v216-external-artifact-population/outcome-audit.json"
    outcome_path = PROJECT_ROOT / "configs/v216-bounded-external-artifact-population-outcome-lock.json"
    results_path = PROJECT_ROOT / "docs/v216-bounded-external-artifact-population-results.md"
    if audit_path.exists() or outcome_path.exists():
        raise RuntimeError("V216 outcome is already audited or frozen")
    config = lock["config_payload"]
    artifacts = {key: PROJECT_ROOT / value for key, value in config["artifacts"].items()}
    retrieval_manifest = json.loads(artifacts["retrievalManifest"].read_text())
    parser_control = json.loads(artifacts["parserControl"].read_text())
    development_public = read_jsonl(artifacts["developmentPublic"])
    development_truth = read_jsonl(artifacts["developmentTruth"])
    protected_public = read_jsonl(artifacts["protectedPublic"])
    protected_truth = read_jsonl(artifacts["protectedTruth"])
    split = json.loads(artifacts["split"].read_text())
    population_manifest = json.loads(artifacts["populationManifest"].read_text())
    summary = json.loads(artifacts["summary"].read_text())
    result = json.loads(artifacts["result"].read_text())
    metrics = score_population(
        retrieval_manifest,
        parser_control,
        development_public + protected_public,
        development_truth + protected_truth,
        split,
        population_manifest,
        config,
        PROJECT_ROOT,
    )
    rebuilt_audit = audit_population(metrics, summary["access"], config)
    raw_checks = []
    for payload in config["payloads"]:
        path = PROJECT_ROOT / payload["rawPath"]
        row = next(row for row in retrieval_manifest["attempts"] if row["payload_id"] == payload["payloadId"])
        raw_checks.append(
            path.is_file()
            and file_sha256(path) == row["sha256"] == summary["raw_payload_sha256"][payload["payloadId"]]
            and path.stat().st_size == payload["expectedByteCount"]
        )
    checks = {
        "design_lock_and_dependencies_exact": dependency_hashes_exact(lock),
        "raw_payloads_and_all_derived_artifacts_are_present_and_frozen": bool(
            all(raw_checks)
            and all(path.is_file() for path in artifacts.values())
            and summary["retrieval_manifest_sha256"] == file_sha256(artifacts["retrievalManifest"])
            and summary["population_manifest_sha256"] == file_sha256(artifacts["populationManifest"])
        ),
        "summary_metrics_access_and_audit_reconstruct_exactly": bool(
            summary["metrics"] == metrics
            and summary["audit"] == rebuilt_audit
            and summary["claim_boundary"] == config["claimBoundary"]
        ),
        "result_reconstructs_and_population_audit_passes": bool(
            result["passed"] == rebuilt_audit["passed"]
            and result["branch"] == rebuilt_audit["branch"]
            and result["decision"] == rebuilt_audit["decision"]
            and rebuilt_audit["passed"]
        ),
        "results_document_exists": results_path.is_file(),
    }
    passed = all(checks.values())
    audit = {
        "schema_version": "216-bounded-external-artifact-population-outcome-audit",
        "experiment": lock["experiment"],
        "passed": passed,
        "branch": rebuilt_audit["branch"],
        "decision": "freeze_verified_V216" if passed else "freeze_failed_V216_verification",
        "checks": checks,
        "metrics": metrics,
    }
    write_json(audit_path, audit)
    if not passed:
        print(json.dumps(audit, indent=2, sort_keys=True))
        raise SystemExit(1)
    dependencies: dict[str, Path] = {
        "design_lock": lock_path,
        "audit": audit_path,
        "retrieval_manifest": artifacts["retrievalManifest"],
        "parser_control": artifacts["parserControl"],
        "development_public": artifacts["developmentPublic"],
        "development_truth": artifacts["developmentTruth"],
        "protected_public": artifacts["protectedPublic"],
        "protected_truth": artifacts["protectedTruth"],
        "split": artifacts["split"],
        "population_manifest": artifacts["populationManifest"],
        "summary": artifacts["summary"],
        "result": artifacts["result"],
        "results_document": results_path,
        "verifier": PROJECT_ROOT / lock["verifier"],
    }
    for payload in config["payloads"]:
        dependencies[f"raw_{payload['payloadId'].lower()}"] = PROJECT_ROOT / payload["rawPath"]
    outcome: dict[str, Any] = {
        "schema_version": "216-bounded-external-artifact-population-outcome-lock",
        "experiment": lock["experiment"],
        "outcome": {
            "passed": True,
            "branch": rebuilt_audit["branch"],
            "decision": rebuilt_audit["decision"],
            "metrics": metrics,
        },
        "authorization": {
            "design_V217_deterministic_external_reconstruction_controls": rebuilt_audit["passed"],
            "open_protected_or_run_model_without_separate_lock": False,
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

