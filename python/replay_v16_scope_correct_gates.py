#!/usr/bin/env python3
"""Replay V15 gates with complete-group applicability restricted to in-fold groups."""

from __future__ import annotations

import json
from pathlib import Path

from v10_protocol import file_sha256


def scope_correct_report(result: dict, minimum: float = 0.5) -> dict:
    original = [
        dict(value) for value in result["primary_transfer_gates"]["checks"]
        if value["name"] != "minimum_fold_complete_intervention_group_accuracy"
    ]
    applicable = []
    not_applicable = []
    for name, value in result["primary_folds"].items():
        if name == "context":
            continue
        full = value["overall"]["ablations"]["fully_predicted"]
        if full["complete_intervention_groups"] > 0:
            applicable.append((name, full["complete_intervention_group_accuracy"]))
        else:
            not_applicable.append(name)
    if len(applicable) != 15 or len(not_applicable) != 11:
        raise RuntimeError("V16 fold topology differs from preregistration")
    minimum_fold, minimum_value = min(applicable, key=lambda item: item[1])
    group = {
        "name": "minimum_applicable_fold_complete_intervention_group_accuracy",
        "value": float(minimum_value),
        "minimum": minimum,
        "passed": minimum_value >= minimum,
        "minimum_fold": minimum_fold,
        "applicable_folds": [name for name, _ in applicable],
        "not_applicable_folds": not_applicable,
    }
    checks = [*original, group]
    return {"passed": all(value["passed"] for value in checks), "checks": checks}


def main() -> None:
    lock_path = Path("configs/v16-scope-correct-replay-lock.json")
    result_path = Path("outputs/v15-full-pipeline/evaluation/result.json")
    audit_path = Path("outputs/v15-full-pipeline/group-scope-audit.json")
    output_path = Path("outputs/v16-scope-correct-replay/result.json")
    if output_path.exists():
        raise RuntimeError(f"V16 result already exists: {output_path}")
    lock = json.loads(lock_path.read_text())
    if file_sha256(result_path) != lock["source"]["v15_result_sha256"]:
        raise RuntimeError("V16 source V15 result changed")
    if file_sha256(audit_path) != lock["source"]["scope_audit_sha256"]:
        raise RuntimeError("V16 source scope audit changed")
    result = json.loads(result_path.read_text())
    for value in result["primary_folds"].values():
        if file_sha256(Path(value["head_artifact"])) != value["head_artifact_sha256"]:
            raise RuntimeError("V16 V15 head artifact changed")
    diagnostics = result["zero_shot_operator_diagnostics"]
    for value in diagnostics.values():
        if file_sha256(Path(value["head_artifact"])) != value["head_artifact_sha256"]:
            raise RuntimeError("V16 V15 diagnostic head artifact changed")
    report = scope_correct_report(result, lock["minimum_complete_group_accuracy"])
    output = {
        "schema_version": 16,
        "experiment": "v16_scope_correct_v15_gate_replay",
        "protocol_lock": str(lock_path),
        "protocol_lock_sha256": file_sha256(lock_path),
        "source_v15_result_sha256": file_sha256(result_path),
        "new_model_fits": 0,
        "new_model_forward_passes": 0,
        "new_predictions": 0,
        "threshold_changes": 0,
        "scope_correct_gates": report,
        "decision": (
            "authorize_separately_locked_final_mechanic_evaluation"
            if report["passed"] else "full_pipeline_remains_blocked"
        ),
        "final_mechanic_accessed": False,
        "lora_authorized": False,
        "data_access": lock["data_access"],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
