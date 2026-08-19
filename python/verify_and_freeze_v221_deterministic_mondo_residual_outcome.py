#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from audit_and_freeze_v167_exact_evidence_gathering_planner import payload_hash
from run_v221_deterministic_mondo_residual import (
    access_ledger,
    dependency_hashes_exact,
    evaluate,
)
from v10_protocol import file_sha256
from v221_deterministic_mondo_residual import audit_evaluation, derive_role_manifest
from v22r2_grounding import PROJECT_ROOT


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def main() -> None:
    lock_path = PROJECT_ROOT / "configs/v221-deterministic-mondo-residual-lock.json"
    lock = json.loads(lock_path.read_text())
    audit_path = PROJECT_ROOT / "outputs/v221-deterministic-mondo-residual/outcome-audit.json"
    outcome_path = PROJECT_ROOT / "configs/v221-deterministic-mondo-residual-outcome-lock.json"
    results_path = PROJECT_ROOT / "docs/v221-deterministic-mondo-residual-results.md"
    if audit_path.exists() or outcome_path.exists():
        raise RuntimeError("V221 outcome is already audited or frozen")
    config = lock["config_payload"]
    artifacts = {key: PROJECT_ROOT / value for key, value in config["artifacts"].items()}
    stored_catalog = json.loads(artifacts["catalogManifest"].read_text())
    stored_residual = json.loads(artifacts["residualManifest"].read_text())
    summary = json.loads(artifacts["summary"].read_text())
    result = json.loads(artifacts["result"].read_text())
    catalog, observations, metrics, residual = evaluate(lock)
    rebuilt = audit_evaluation(metrics, catalog, access_ledger(), config)
    population_manifest = json.loads((PROJECT_ROOT / config["inputContract"]["populationManifest"]).read_text())
    rebuilt_roles = derive_role_manifest(population_manifest["development_group_ids"], config)
    stored_roles = json.loads(artifacts["roleManifest"].read_text())
    observation_lines = [json.dumps(value, sort_keys=True) + "\n" for value in observations]
    observed_observation_hash = __import__("hashlib").sha256("".join(observation_lines).encode("utf-8")).hexdigest()
    checks = {
        "design_lock_dependencies_and_hash_only_protected_contract_are_exact": bool(
            dependency_hashes_exact(lock)
            and file_sha256(PROJECT_ROOT / config["inputContract"]["protectedPublic"]) == config["inputContract"]["protectedPublicSha256"]
            and file_sha256(PROJECT_ROOT / config["inputContract"]["protectedTruth"]) == config["inputContract"]["protectedTruthSha256"]
        ),
        "role_catalog_observations_and_residual_reconstruct_exactly": bool(
            stored_roles == rebuilt_roles
            and stored_catalog == catalog
            and observed_observation_hash == file_sha256(artifacts["observations"])
            and stored_residual == residual
        ),
        "summary_metrics_access_and_audit_reconstruct_exactly": bool(
            summary["metrics"] == metrics and summary["access"] == access_ledger()
            and summary["audit"] == rebuilt and summary["claim_boundary"] == config["claimBoundary"]
            and summary["role_manifest_sha256"] == file_sha256(artifacts["roleManifest"])
            and summary["catalog_manifest_sha256"] == file_sha256(artifacts["catalogManifest"])
            and summary["observations_sha256"] == file_sha256(artifacts["observations"])
            and summary["residual_manifest_sha256"] == file_sha256(artifacts["residualManifest"])
        ),
        "result_reconstructs_for_integrity_residual_or_sufficiency_branch": bool(
            result["passed"] == rebuilt["passed"] and result["branch"] == rebuilt["branch"]
            and result["decision"] == rebuilt["decision"]
            and result["residual_evaluation_group_count"] == metrics["residual_evaluation_group_count"]
        ),
        "results_document_exists": results_path.is_file(),
    }
    passed = all(checks.values())
    audit = {
        "schema_version": "221-deterministic-mondo-residual-outcome-audit",
        "experiment": lock["experiment"], "passed": passed,
        "scientific_passed": rebuilt["passed"], "branch": rebuilt["branch"],
        "decision": "freeze_verified_V221" if passed else "freeze_failed_V221_verification",
        "checks": checks,
        "protected_JSONL_body_load_count": 0,
        "metrics": metrics,
    }
    write_json(audit_path, audit)
    if not passed:
        print(json.dumps(audit, indent=2, sort_keys=True))
        raise SystemExit(1)
    dependencies = {
        "design_lock": lock_path, "audit": audit_path, "role_manifest": artifacts["roleManifest"],
        "catalog_manifest": artifacts["catalogManifest"], "observations": artifacts["observations"],
        "residual_manifest": artifacts["residualManifest"], "summary": artifacts["summary"],
        "result": artifacts["result"], "results_document": results_path,
        "verifier": PROJECT_ROOT / lock["verifier"],
    }
    model_design = bool(rebuilt["passed"] and metrics["model_eligible_residual"])
    outcome: dict[str, Any] = {
        "schema_version": "221-deterministic-mondo-residual-outcome-lock",
        "experiment": lock["experiment"],
        "outcome": {
            "verification_passed": True, "scientific_passed": rebuilt["passed"],
            "branch": rebuilt["branch"], "decision": rebuilt["decision"],
            "residual_evaluation_group_count": metrics["residual_evaluation_group_count"],
            "model_eligible_residual": metrics["model_eligible_residual"],
            "protected_JSONL_body_load_count": 0,
            "metrics": metrics,
        },
        "authorization": {
            "design_one_local_model_candidate_study": model_design,
            "open_protected_or_run_model": False,
            "register_mutate_service_act_execute": False,
        },
    }
    for key, path in dependencies.items():
        outcome[key] = str(path.relative_to(PROJECT_ROOT))
        outcome[f"{key}_sha256"] = file_sha256(path)
    outcome["protected_hashes"] = {
        config["inputContract"]["protectedPublic"]: config["inputContract"]["protectedPublicSha256"],
        config["inputContract"]["protectedTruth"]: config["inputContract"]["protectedTruthSha256"],
    }
    outcome["lock_payload_sha256"] = payload_hash(outcome)
    write_json(outcome_path, outcome)
    print(json.dumps(audit, indent=2, sort_keys=True))
    print(json.dumps({"outcome_lock": str(outcome_path.relative_to(PROJECT_ROOT)), "sha256": file_sha256(outcome_path)}, indent=2))


if __name__ == "__main__":
    main()
