#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from run_v221_deterministic_mondo_residual import access_ledger, dependency_hashes_exact, evaluate, write_json, write_jsonl
from v10_protocol import file_sha256
from v221_deterministic_mondo_residual import audit_evaluation
from v22r2_grounding import PROJECT_ROOT


def repaired_lock(base_lock: dict[str, Any], repair: dict[str, Any]) -> dict[str, Any]:
    value = json.loads(json.dumps(base_lock))
    value["config_payload"]["parserDesign"] = repair["repair"]["parserDesign"]
    return value


def main() -> None:
    repair_lock_path = PROJECT_ROOT / "configs/v221r1-parser-config-repair-lock.json"
    repair_lock = json.loads(repair_lock_path.read_text())
    base_lock = json.loads((PROJECT_ROOT / repair_lock["parent_V221_design"]).read_text())
    if not dependency_hashes_exact(base_lock):
        raise RuntimeError("V221 design lock or dependency hash mismatch")
    if not repair_lock["authorization"]["run_one_repaired_V221_development_evaluation"]:
        raise RuntimeError("V221r1 evaluation is not authorized")
    repair = repair_lock["repair_config_payload"]
    config = base_lock["config_payload"]
    artifacts = {key: PROJECT_ROOT / value for key, value in repair["artifacts"].items()}
    if any(path.exists() for path in artifacts.values()):
        raise RuntimeError("V221r1 output already exists")
    runtime_lock = repaired_lock(base_lock, repair)
    catalog, observations, metrics, residual = evaluate(runtime_lock)
    access = access_ledger()
    audit = audit_evaluation(metrics, catalog, access, runtime_lock["config_payload"])
    write_json(artifacts["catalogManifest"], catalog)
    write_jsonl(artifacts["observations"], observations)
    write_json(artifacts["residualManifest"], residual)
    summary = {
        "schema_version": "221r1-parser-config-repair-summary",
        "experiment": repair["experiment"],
        "base_experiment": config["experiment"],
        "claim_boundary": config["claimBoundary"],
        "repair_operation": repair["repair"]["operation"],
        "prior_failed_development_public_load_count": repair["failureBoundary"]["developmentPublicPriorLoadCount"],
        "prior_failed_development_truth_load_count": repair["failureBoundary"]["developmentTruthPriorLoadCount"],
        "candidate_method_evaluation_count_before_repair": repair["failureBoundary"]["candidateMethodEvaluationCount"],
        "catalog_manifest_sha256": file_sha256(artifacts["catalogManifest"]),
        "observations_sha256": file_sha256(artifacts["observations"]),
        "residual_manifest_sha256": file_sha256(artifacts["residualManifest"]),
        "metrics": metrics, "access": access, "audit": audit,
    }
    model_design = bool(audit["passed"] and metrics["model_eligible_residual"])
    result = {
        "schema_version": "221r1-parser-config-repair-result",
        "experiment": repair["experiment"], "base_experiment": config["experiment"],
        "passed": audit["passed"], "branch": audit["branch"], "decision": audit["decision"],
        "residual_evaluation_group_count": metrics["residual_evaluation_group_count"],
        "authorization": {
            "design_one_local_model_candidate_study": model_design,
            "run_model_or_open_protected": False,
            "register_mutate_service_act_execute": False
        }
    }
    write_json(artifacts["summary"], summary)
    write_json(artifacts["result"], result)
    print(json.dumps(result, indent=2, sort_keys=True))
    print(json.dumps({key: value for key, value in metrics.items() if key not in {"cells", "primary_evaluation_strata", "residual_evaluation_group_ids"}}, indent=2, sort_keys=True))
    if not audit["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
