#!/usr/bin/env python3
"""Audit whether V15's intervention-group gate stayed within each fold's evaluation scope."""

from __future__ import annotations

import json
from pathlib import Path

from v10_protocol import file_sha256


def main() -> None:
    result_path = Path("outputs/v15-full-pipeline/evaluation/result.json")
    output_path = Path("outputs/v15-full-pipeline/group-scope-audit.json")
    result = json.loads(result_path.read_text())
    folds = {}
    applicable = []
    expanded = []
    for name, value in result["primary_folds"].items():
        if name == "context":
            continue
        full = value["overall"]["ablations"]["fully_predicted"]
        in_fold_groups = full["complete_intervention_groups"]
        in_fold_accuracy = full["complete_intervention_group_accuracy"]
        scope = value["group_scope"]
        scope_expanded = scope["records"] != value["overall"]["records"]
        folds[name] = {
            "kind": value["kind"],
            "evaluation_records": value["overall"]["records"],
            "in_fold_complete_groups": in_fold_groups,
            "in_fold_complete_group_accuracy": in_fold_accuracy,
            "configured_group_scope_records": scope["records"],
            "configured_group_scope_complete_groups": scope["complete_intervention_groups"],
            "configured_group_scope_accuracy": scope["complete_intervention_group_accuracy"],
            "scope_expanded_beyond_evaluation": scope_expanded,
            "topologically_applicable": in_fold_groups > 0,
        }
        if in_fold_groups > 0:
            applicable.append((name, in_fold_accuracy))
        if scope_expanded:
            expanded.append(name)
    minimum_name, minimum_value = min(applicable, key=lambda item: item[1])
    original = next(
        item for item in result["primary_transfer_gates"]["checks"]
        if item["name"] == "minimum_fold_complete_intervention_group_accuracy"
    )
    audit = {
        "schema_version": 15,
        "experiment": "v15_post_result_intervention_group_scope_audit",
        "v15_result_sha256": file_sha256(result_path),
        "original_locked_gate": original,
        "expanded_scope_folds": expanded,
        "topologically_applicable_folds": [name for name, _ in applicable],
        "scope_correct_minimum": {
            "fold": minimum_name,
            "value": minimum_value,
            "minimum": original["minimum"],
            "passed": minimum_value >= original["minimum"],
        },
        "operator_multiway_evaluation_is_perfect": (
            result["primary_folds"]["operator:multiway_partition"]["overall"]["ablations"]["fully_predicted"]
            ["allowed_values_accuracy"] == 1.0
        ),
        "folds": folds,
        "v15_original_decision_unchanged": result["decision"],
        "final_mechanic_access_remains_closed": True,
        "decision": "preregister_scope_correct_exact_replay_before_final_access",
        "data_access": result["data_access"],
    }
    output_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    print(json.dumps(audit, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
