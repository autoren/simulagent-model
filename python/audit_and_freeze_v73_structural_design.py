#!/usr/bin/env python3
"""Freeze the V73 structural and development designs before implementation."""
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
    source_lock_path = PROJECT_ROOT / "configs/v73-active-sensing-source-lock.json"
    structural_config_path = (
        PROJECT_ROOT / "configs/v73-active-sensing-structural-design.json"
    )
    structural_plan_path = (
        PROJECT_ROOT / "docs/v73-active-sensing-structural-design-plan.md"
    )
    evaluation_config_path = (
        PROJECT_ROOT / "configs/v73-active-sensing-development-evaluation.json"
    )
    evaluation_plan_path = (
        PROJECT_ROOT / "docs/v73-active-sensing-development-evaluation-plan.md"
    )
    auditor_path = PROJECT_ROOT / "python/audit_and_freeze_v73_structural_design.py"
    lock_path = PROJECT_ROOT / "configs/v73-active-sensing-design-lock.json"
    if lock_path.exists():
        raise RuntimeError("V73 structural design is already frozen")

    source_lock = json.loads(source_lock_path.read_text())
    source_payload = {
        key: value for key, value in source_lock.items() if key != "lock_payload_sha256"
    }
    structural = json.loads(structural_config_path.read_text())
    evaluation = json.loads(evaluation_config_path.read_text())
    errors: list[str] = []
    source_authorization_ok = bool(
        payload_hash(source_payload) == source_lock["lock_payload_sha256"]
        and source_lock["authorization"]["implement_adapter_and_structural_tests"]
        and source_lock["authorization"][
            "run_preregistered_structural_dominance_audit"
        ]
        and not source_lock["authorization"][
            "compute_exact_BA_MAP_PS_or_myopic_outcomes"
        ]
    )
    if not source_authorization_ok:
        errors.append("V73 source lock does not authorize structural implementation")

    future_paths = (
        PROJECT_ROOT / "python/v73_imprl_maintenance_source.py",
        PROJECT_ROOT / "python/test_v73_imprl_maintenance_source.py",
        PROJECT_ROOT / "python/audit_and_freeze_v73_structural_feasibility.py",
        PROJECT_ROOT / "python/evaluate_v73_imprl_development.py",
    )
    preimplementation_ok = not any(path.exists() for path in future_paths)
    if not preimplementation_ok:
        errors.append("V73 design was not frozen before implementation files existed")

    design_ok = bool(
        structural["horizonActions"] == 5
        and structural["structuralGates"]["requiredStructuralUnitTests"] == 10
        and structural["structuralGates"]
        ["minimumFixedAdaptivePolicyOverBestOpenLoopNormalizedAdvantage"]
        == 0.005
        and structural["structuralGates"]["maximumExactBayesAdaptiveCalls"] == 0
        and evaluation["horizonActions"] == 5
        and evaluation["gates"]["requiredExactRootAction"] == "calibrate_beacon"
        and evaluation["gates"]["requiredMAPRootAction"] == "inspect_target"
        and evaluation["gates"]["minimumNormalizedMAPRegret"] == 0.005
        and evaluation["decisionRule"]["noRetrospectiveChanges"] is True
    )
    if not design_ok:
        errors.append("V73 structural or evaluation preregistration drifted")

    if errors:
        print(json.dumps({"passed": False, "errors": errors}, indent=2))
        raise SystemExit(1)

    lock = {
        "schema_version": "73-active-sensing-design",
        "experiment": "v73_structural_and_development_design_lock",
        "source_lock": str(source_lock_path.relative_to(PROJECT_ROOT)),
        "source_lock_sha256": file_sha256(source_lock_path),
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
            "modify_source_blueprint_structural_or_evaluation_design": False,
            "implement_and_test_source_grounded_adapter": True,
            "run_structural_dominance_audit_once": True,
            "run_exact_development_evaluation": False,
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
