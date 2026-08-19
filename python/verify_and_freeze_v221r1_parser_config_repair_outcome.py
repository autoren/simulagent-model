#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from audit_and_freeze_v165_factored_ontology_identifiability_population import valid_lock
from audit_and_freeze_v167_exact_evidence_gathering_planner import payload_hash
from run_v221_deterministic_mondo_residual import access_ledger, dependency_hashes_exact, evaluate
from run_v221r1_parser_config_repair import repaired_lock
from v10_protocol import file_sha256
from v221_deterministic_mondo_residual import audit_evaluation
from v22r2_grounding import PROJECT_ROOT


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def main() -> None:
    repair_lock_path = PROJECT_ROOT / "configs/v221r1-parser-config-repair-lock.json"
    repair_lock = json.loads(repair_lock_path.read_text())
    audit_path = PROJECT_ROOT / "outputs/v221r1-parser-config-repair/outcome-audit.json"
    outcome_path = PROJECT_ROOT / "configs/v221r1-parser-config-repair-outcome-lock.json"
    results_path = PROJECT_ROOT / "docs/v221r1-parser-config-repair-results.md"
    if audit_path.exists() or outcome_path.exists():
        raise RuntimeError("V221r1 outcome is already audited or frozen")
    repair = repair_lock["repair_config_payload"]
    base_lock = json.loads((PROJECT_ROOT / repair_lock["parent_V221_design"]).read_text())
    runtime_lock = repaired_lock(base_lock, repair)
    artifacts = {key: PROJECT_ROOT / value for key, value in repair["artifacts"].items()}
    stored_catalog = json.loads(artifacts["catalogManifest"].read_text())
    stored_residual = json.loads(artifacts["residualManifest"].read_text())
    summary = json.loads(artifacts["summary"].read_text())
    result = json.loads(artifacts["result"].read_text())
    catalog, observations, metrics, residual = evaluate(runtime_lock)
    rebuilt = audit_evaluation(metrics, catalog, access_ledger(), runtime_lock["config_payload"])
    observation_hash = hashlib.sha256("".join(json.dumps(row, sort_keys=True) + "\n" for row in observations).encode("utf-8")).hexdigest()
    inputs = base_lock["config_payload"]["inputContract"]
    checks = {
        "repair_and_base_design_locks_are_exact_and_protected_remains_hash_only": bool(
            valid_lock(repair_lock) and dependency_hashes_exact(base_lock)
            and file_sha256(PROJECT_ROOT / inputs["protectedPublic"]) == inputs["protectedPublicSha256"]
            and file_sha256(PROJECT_ROOT / inputs["protectedTruth"]) == inputs["protectedTruthSha256"]
        ),
        "catalog_observations_and_residual_reconstruct_under_exact_parser_injection": bool(
            stored_catalog == catalog and stored_residual == residual
            and observation_hash == file_sha256(artifacts["observations"])
        ),
        "summary_metrics_access_and_scientific_audit_reconstruct_exactly": bool(
            summary["metrics"] == metrics and summary["access"] == access_ledger()
            and summary["audit"] == rebuilt
            and summary["candidate_method_evaluation_count_before_repair"] == 0
            and summary["catalog_manifest_sha256"] == file_sha256(artifacts["catalogManifest"])
            and summary["observations_sha256"] == file_sha256(artifacts["observations"])
            and summary["residual_manifest_sha256"] == file_sha256(artifacts["residualManifest"])
        ),
        "result_reconstructs_for_negative_residual_or_sufficiency_branch": bool(
            result["passed"] == rebuilt["passed"] and result["branch"] == rebuilt["branch"]
            and result["decision"] == rebuilt["decision"]
            and result["residual_evaluation_group_count"] == metrics["residual_evaluation_group_count"]
        ),
        "results_document_exists": results_path.is_file(),
    }
    passed = all(checks.values())
    audit = {
        "schema_version": "221r1-parser-config-repair-outcome-audit",
        "experiment": repair["experiment"], "passed": passed,
        "scientific_passed": rebuilt["passed"], "branch": rebuilt["branch"],
        "decision": "freeze_verified_V221r1" if passed else "freeze_failed_V221r1_verification",
        "checks": checks, "protected_JSONL_body_load_count": 0, "metrics": metrics,
    }
    write_json(audit_path, audit)
    if not passed:
        print(json.dumps(audit, indent=2, sort_keys=True))
        raise SystemExit(1)
    dependencies = {
        "repair_lock": repair_lock_path, "base_design_lock": PROJECT_ROOT / repair_lock["parent_V221_design"],
        "audit": audit_path, "catalog_manifest": artifacts["catalogManifest"],
        "observations": artifacts["observations"], "residual_manifest": artifacts["residualManifest"],
        "summary": artifacts["summary"], "result": artifacts["result"],
        "results_document": results_path, "verifier": PROJECT_ROOT / repair_lock["verifier"],
    }
    model_design = bool(rebuilt["passed"] and metrics["model_eligible_residual"])
    outcome: dict[str, Any] = {
        "schema_version": "221r1-parser-config-repair-outcome-lock",
        "experiment": repair["experiment"],
        "outcome": {
            "verification_passed": True, "scientific_passed": rebuilt["passed"],
            "branch": rebuilt["branch"], "decision": rebuilt["decision"],
            "residual_evaluation_group_count": metrics["residual_evaluation_group_count"],
            "model_eligible_residual": metrics["model_eligible_residual"],
            "protected_JSONL_body_load_count": 0, "metrics": metrics,
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
        inputs["protectedPublic"]: inputs["protectedPublicSha256"],
        inputs["protectedTruth"]: inputs["protectedTruthSha256"],
    }
    outcome["lock_payload_sha256"] = payload_hash(outcome)
    write_json(outcome_path, outcome)
    print(json.dumps(audit, indent=2, sort_keys=True))
    print(json.dumps({"outcome_lock": str(outcome_path.relative_to(PROJECT_ROOT)), "sha256": file_sha256(outcome_path)}, indent=2))


if __name__ == "__main__":
    main()
