#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from audit_and_freeze_v167_exact_evidence_gathering_planner import payload_hash
from run_v213_fresh_programmatic_concept_population import dependency_hashes_exact
from v10_protocol import file_sha256
from v213_fresh_programmatic_concept_population import (
    audit_population,
    generate_population,
    project_public_blueprints,
    score_population,
)
from v22r2_grounding import PROJECT_ROOT


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def main() -> None:
    lock_path = PROJECT_ROOT / "configs/v213-fresh-programmatic-concept-population-lock.json"
    lock = json.loads(lock_path.read_text())
    audit_path = PROJECT_ROOT / "outputs/v213-programmatic-concept-population/outcome-audit.json"
    outcome_path = PROJECT_ROOT / "configs/v213-fresh-programmatic-concept-population-outcome-lock.json"
    results_path = PROJECT_ROOT / "docs/v213-fresh-programmatic-concept-population-results.md"
    if audit_path.exists() or outcome_path.exists():
        raise RuntimeError("V213 outcome is already audited or frozen")
    config = lock["config_payload"]
    artifacts = {key: PROJECT_ROOT / value for key, value in config["artifacts"].items()}
    semantics = json.loads((PROJECT_ROOT / lock["parent_public_semantics"]).read_text())
    rebuilt_blueprints, rebuilt_truth, rebuilt_split, rebuilt_manifest = generate_population(config, semantics)
    rebuilt_public = project_public_blueprints(rebuilt_blueprints)
    stored_manifest = json.loads(artifacts["generatorManifest"].read_text())
    stored_blueprints = read_jsonl(artifacts["publicBlueprints"])
    stored_truth = read_jsonl(artifacts["sealedTruth"])
    stored_split = json.loads(artifacts["split"].read_text())
    stored_public = read_jsonl(artifacts["publicRecords"])
    freeze = json.loads(artifacts["publicProjectionFreeze"].read_text())
    summary = json.loads(artifacts["summary"].read_text())
    result = json.loads(artifacts["result"].read_text())
    parent_public = read_jsonl(PROJECT_ROOT / lock["parent_public_cases"])
    metrics = score_population(
        rebuilt_blueprints,
        rebuilt_public,
        rebuilt_truth,
        rebuilt_split,
        semantics,
        parent_public,
        freeze,
        config,
    )
    rebuilt_audit = audit_population(metrics, summary["access"], config)
    checks = {
        "design_lock_and_dependencies_exact": dependency_hashes_exact(lock),
        "generator_manifest_blueprints_truth_and_split_reconstruct_exactly": bool(
            stored_manifest == rebuilt_manifest
            and stored_blueprints == rebuilt_blueprints
            and stored_truth == rebuilt_truth
            and stored_split == rebuilt_split
        ),
        "public_projection_reconstructs_exactly": stored_public == rebuilt_public,
        "public_projection_freeze_hashes_and_firewall_exact": bool(
            freeze["design_lock_sha256"] == file_sha256(lock_path)
            and freeze["parent_outcome_lock_sha256"] == file_sha256(PROJECT_ROOT / lock["parent_V212_outcome"])
            and freeze["parent_public_semantics_sha256"] == file_sha256(PROJECT_ROOT / lock["parent_public_semantics"])
            and freeze["generator_manifest_sha256"] == file_sha256(artifacts["generatorManifest"])
            and freeze["public_blueprints_sha256"] == file_sha256(artifacts["publicBlueprints"])
            and freeze["sealed_truth_sha256"] == file_sha256(artifacts["sealedTruth"])
            and freeze["split_sha256"] == file_sha256(artifacts["split"])
            and freeze["public_records_sha256"] == file_sha256(artifacts["publicRecords"])
            and freeze["public_projection_frozen_before_truth_join"]
            and not freeze["sealed_truth_joined_before_public_projection_freeze"]
            and freeze["public_projection_worker_sealed_truth_path_count"] == 0
            and freeze["public_projection_worker_forbidden_token_count"] == 0
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
        "schema_version": "213-fresh-programmatic-concept-population-outcome-audit",
        "experiment": lock["experiment"],
        "passed": passed,
        "branch": rebuilt_audit["branch"],
        "decision": "freeze_verified_V213" if passed else "freeze_failed_V213_verification",
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
        "generator_manifest": artifacts["generatorManifest"],
        "public_blueprints": artifacts["publicBlueprints"],
        "sealed_truth": artifacts["sealedTruth"],
        "split": artifacts["split"],
        "public_records": artifacts["publicRecords"],
        "public_projection_freeze": artifacts["publicProjectionFreeze"],
        "summary": artifacts["summary"],
        "result": artifacts["result"],
        "results_document": results_path,
        "verifier": PROJECT_ROOT / lock["verifier"],
    }
    outcome: dict[str, Any] = {
        "schema_version": "213-fresh-programmatic-concept-population-outcome-lock",
        "experiment": lock["experiment"],
        "outcome": {
            "passed": True,
            "branch": rebuilt_audit["branch"],
            "decision": rebuilt_audit["decision"],
            "metrics": metrics,
        },
        "authorization": {
            "design_V214_deterministic_candidate_and_version_space_controls": rebuilt_audit["branch"] == "V214_DETERMINISTIC_CONTROL_DESIGN_ELIGIBLE",
            "run_V214_without_separate_lock": False,
            "open_protected_downstream_or_run_model": False,
            "read_external_payload_register_mutate_call_act_or_execute": False,
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
