#!/usr/bin/env python3
"""Freeze V74 structural and development designs before adapter implementation."""
from __future__ import annotations

import hashlib
import json
from typing import Any

from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT


def payload_hash(value: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def main() -> None:
    economic_lock_path = PROJECT_ROOT / "configs/v74-active-sensing-economic-lock.json"
    structural_config_path = PROJECT_ROOT / "configs/v74-active-sensing-structural-design.json"
    structural_plan_path = PROJECT_ROOT / "docs/v74-active-sensing-structural-design-plan.md"
    evaluation_config_path = PROJECT_ROOT / "configs/v74-active-sensing-development-evaluation.json"
    evaluation_plan_path = PROJECT_ROOT / "docs/v74-active-sensing-development-evaluation-plan.md"
    auditor_path = PROJECT_ROOT / "python/audit_and_freeze_v74_design.py"
    lock_path = PROJECT_ROOT / "configs/v74-active-sensing-design-lock.json"
    if lock_path.exists():
        raise RuntimeError("V74 design is already frozen")

    economic_lock = json.loads(economic_lock_path.read_text())
    economic_payload = {
        key: value for key, value in economic_lock.items() if key != "lock_payload_sha256"
    }
    structural = json.loads(structural_config_path.read_text())
    evaluation = json.loads(evaluation_config_path.read_text())
    errors: list[str] = []
    authorization_ok = bool(
        payload_hash(economic_payload) == economic_lock["lock_payload_sha256"]
        and economic_lock["outcome"]["passed_all_economic_gates"]
        and economic_lock["authorization"][
            "preregister_adapter_structural_and_development_design"
        ]
        and not economic_lock["authorization"]["implement_adapter_and_structural_tests"]
        and not economic_lock["authorization"]["run_optimal_planners"]
    )
    if not authorization_ok:
        errors.append("V74 economic lock does not authorize design preregistration")

    future_paths = (
        PROJECT_ROOT / "python/v74_pomdppy_tiger_source.py",
        PROJECT_ROOT / "python/test_v74_pomdppy_tiger_source.py",
        PROJECT_ROOT / "python/audit_and_freeze_v74_structural_feasibility.py",
        PROJECT_ROOT / "python/evaluate_v74_pomdppy_tiger_development.py",
        PROJECT_ROOT / "python/audit_and_freeze_v74_development_evaluator.py",
    )
    preimplementation_ok = not any(path.exists() for path in future_paths)
    if not preimplementation_ok:
        errors.append("V74 designs were not frozen before implementation files existed")

    sgates = structural["structuralGates"]
    egates = evaluation["gates"]
    design_ok = bool(
        structural["horizonActions"] == 3
        and sgates["requiredStructuralUnitTests"] == 10
        and sgates["minimumFixedAdaptivePolicyOverBestOpenLoopNormalizedAdvantage"] == 0.015
        and sgates["minimumNormalizedMarginAboveEconomicThreshold"] == 0.005
        and sgates["maximumExactBayesAdaptiveCalls"] == 0
        and evaluation["horizonActions"] == 3
        and egates["requiredExactRootAction"] == "calibrate_beacon"
        and egates["requiredMAPRootAction"] == "listen_target"
        and egates["minimumNormalizedMAPRegret"] == 0.1
        and egates["minimumNormalizedPosteriorSamplingRegret"] == 0.1
        and egates["minimumNormalizedExactOverOpenLoopAdvantage"] == 0.015
        and evaluation["decisionRule"]["noRetrospectiveChanges"] is True
        and not evaluation["claimBoundary"]["unchangedExternalEnvironment"]
    )
    if not design_ok:
        errors.append("V74 structural or evaluation preregistration drifted")

    if errors:
        print(json.dumps({"passed": False, "errors": errors}, indent=2))
        raise SystemExit(1)

    lock = {
        "schema_version": "74-active-sensing-design",
        "experiment": "v74_structural_and_development_design_lock",
        "economic_lock": str(economic_lock_path.relative_to(PROJECT_ROOT)),
        "economic_lock_sha256": file_sha256(economic_lock_path),
        "structural_config": str(structural_config_path.relative_to(PROJECT_ROOT)),
        "structural_config_sha256": file_sha256(structural_config_path),
        "structural_plan": str(structural_plan_path.relative_to(PROJECT_ROOT)),
        "structural_plan_sha256": file_sha256(structural_plan_path),
        "evaluation_config": str(evaluation_config_path.relative_to(PROJECT_ROOT)),
        "evaluation_config_sha256": file_sha256(evaluation_config_path),
        "evaluation_plan": str(evaluation_plan_path.relative_to(PROJECT_ROOT)),
        "evaluation_plan_sha256": file_sha256(evaluation_plan_path),
        "auditor": str(auditor_path.relative_to(PROJECT_ROOT)),
        "auditor_sha256": file_sha256(auditor_path),
        "preimplementation_file_count": 0,
        "authorization": {
            "modify_source_economic_structural_or_evaluation_design": False,
            "implement_and_test_source_grounded_adapter": True,
            "run_structural_audit_once": True,
            "run_optimal_planner_outcomes": False,
            "select_or_inspect_confirmation_sources": False,
            "run_SMC2": False,
            "human_data": False,
            "model_access": False,
            "adapter_training": False,
        },
    }
    lock["lock_payload_sha256"] = payload_hash(lock)
    lock_path.write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"passed": True, "lock": str(lock_path), "sha256": file_sha256(lock_path)}, indent=2))


if __name__ == "__main__":
    main()
