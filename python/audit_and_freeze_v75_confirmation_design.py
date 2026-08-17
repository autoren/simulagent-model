#!/usr/bin/env python3
"""Freeze V75 structural and one-shot evaluation designs before adapter code."""
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
    economic_lock_path = PROJECT_ROOT / "configs/v75-active-sensing-confirmation-economic-lock.json"
    structural_path = PROJECT_ROOT / "configs/v75-active-sensing-confirmation-structural.json"
    evaluation_path = PROJECT_ROOT / "configs/v75-active-sensing-confirmation-evaluation.json"
    plan_path = PROJECT_ROOT / "docs/v75-active-sensing-confirmation-design-plan.md"
    auditor_path = PROJECT_ROOT / "python/audit_and_freeze_v75_confirmation_design.py"
    lock_path = PROJECT_ROOT / "configs/v75-active-sensing-confirmation-design-lock.json"
    if lock_path.exists():
        raise RuntimeError("V75 confirmation design is already frozen")

    economic_lock = json.loads(economic_lock_path.read_text())
    economic_payload = {
        key: value for key, value in economic_lock.items() if key != "lock_payload_sha256"
    }
    structural = json.loads(structural_path.read_text())
    evaluation = json.loads(evaluation_path.read_text())
    errors: list[str] = []
    authorization_ok = bool(
        payload_hash(economic_payload) == economic_lock["lock_payload_sha256"]
        and economic_lock["outcome"]["passed_all_economic_gates"]
        and economic_lock["authorization"]["preregister_exact_replication_design"]
        and not economic_lock["authorization"]["implement_adapter_and_structural_tests"]
        and not economic_lock["authorization"]["run_policy_outcomes"]
    )
    if not authorization_ok:
        errors.append("V75 economic lock does not authorize design preregistration")

    future_paths = (
        PROJECT_ROOT / "python/v75_nova_paint_source.py",
        PROJECT_ROOT / "python/test_v75_nova_paint_source.py",
        PROJECT_ROOT / "python/audit_and_freeze_v75_confirmation_structural.py",
        PROJECT_ROOT / "python/evaluate_v75_nova_paint_confirmation.py",
        PROJECT_ROOT / "python/audit_and_freeze_v75_confirmation_evaluator.py",
    )
    preimplementation_ok = not any(path.exists() for path in future_paths)
    if not preimplementation_ok:
        errors.append("V75 designs were not frozen before implementation files existed")

    sgates = structural["structuralGates"]
    egates = evaluation["gates"]
    design_ok = bool(
        structural["horizonActions"] == 4
        and sgates["requiredStructuralUnitTests"] == 10
        and sgates["minimumFixedAdaptivePolicyOverBestOpenLoopNormalizedAdvantage"] == 0.015
        and sgates["maximumExactBayesAdaptiveCalls"] == 0
        and evaluation["horizonActions"] == 4
        and egates["requiredExactRootAction"] == "calibrate_beacon"
        and egates["requiredExactRootOptimalActions"]
        == ["calibrate_beacon", "inspect_target"]
        and egates["requiredMAPRootAction"] == "inspect_target"
        and egates["minimumNormalizedMAPRegret"] == 0.015
        and egates["minimumNormalizedPosteriorSamplingRegret"] == 0.015
        and egates["maximumEvaluationAttempts"] == 1
        and evaluation["decisionRule"]["noRetrospectiveChanges"] is True
        and not evaluation["claimBoundary"]["sourceDiscoveryCleanConfirmation"]
    )
    if not design_ok:
        errors.append("V75 structural or evaluation preregistration drifted")

    if errors:
        print(json.dumps({"passed": False, "errors": errors}, indent=2))
        raise SystemExit(1)

    lock = {
        "schema_version": "75-active-sensing-confirmation-design",
        "experiment": "v75_structural_and_replication_design_lock",
        "economic_lock": str(economic_lock_path.relative_to(PROJECT_ROOT)),
        "economic_lock_sha256": file_sha256(economic_lock_path),
        "structural_config": str(structural_path.relative_to(PROJECT_ROOT)),
        "structural_config_sha256": file_sha256(structural_path),
        "evaluation_config": str(evaluation_path.relative_to(PROJECT_ROOT)),
        "evaluation_config_sha256": file_sha256(evaluation_path),
        "plan": str(plan_path.relative_to(PROJECT_ROOT)),
        "plan_sha256": file_sha256(plan_path),
        "auditor": str(auditor_path.relative_to(PROJECT_ROOT)),
        "auditor_sha256": file_sha256(auditor_path),
        "preimplementation_file_count": 0,
        "authorization": {
            "modify_source_economic_structural_or_evaluation_design": False,
            "implement_and_test_source_grounded_adapter": True,
            "run_structural_audit_once": True,
            "run_policy_outcomes": False,
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
