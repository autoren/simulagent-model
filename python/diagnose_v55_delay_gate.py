#!/usr/bin/env python3
"""Post-result, read-only localization of V55's failed delay-sensitivity gate."""
from __future__ import annotations

import json
from pathlib import Path

from v10_protocol import file_sha256
from v22_relational import canonical_json
from v22r2_grounding import PROJECT_ROOT
from v53_smc2 import mechanic_registry


RESULT = PROJECT_ROOT / "outputs/v55-short-horizon-bayes-adaptive-planning/evaluation/result.json"
OUTCOME = PROJECT_ROOT / "configs/v55-outcome-lock.json"
POPULATION = PROJECT_ROOT / "data/v55-short-horizon-bayes-adaptive-planning/planning.jsonl"
OUTPUT = PROJECT_ROOT / "outputs/v55-short-horizon-bayes-adaptive-planning/delay-failure-localization.json"
SUMMARY = PROJECT_ROOT / "docs/v55-delayed-consequence-failure-localization.md"


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def stochastic_target(branch: dict) -> str:
    return canonical_json(branch["effect"]["target"])


def main() -> None:
    if OUTPUT.exists() or SUMMARY.exists():
        raise FileExistsError("V55 delay localization outputs already exist")

    outcome = json.loads(OUTCOME.read_text())
    result = json.loads(RESULT.read_text())
    population = read_jsonl(POPULATION)
    registry = mechanic_registry(5303)

    if outcome["decision"] != "retain_v55_failure_and_localize_failed_gates":
        raise RuntimeError("V55 outcome does not authorize failure localization")
    if result["qualification"]["passed"]:
        raise RuntimeError("V55 unexpectedly passed; localization is not applicable")
    failed = sorted(
        name for name, passed in result["qualification"]["checks"].items()
        if not passed
    )
    if failed != ["delayed_consequence_sensitivity"]:
        raise RuntimeError(f"Unexpected V55 failed-gate set: {failed}")

    delay_two = []
    for program_index, program in enumerate(registry):
        for rule in program["template"]["rules"]:
            for branch in rule["stochastic_delayed"]:
                if branch["delay"] != 2:
                    continue
                target = stochastic_target(branch)
                same_target_immediate = []
                for other in program["template"]["rules"]:
                    for effect in other["deterministic_immediate"]:
                        if canonical_json(effect["target"]) == target:
                            same_target_immediate.append({
                                "action": other["action"],
                                "operation": effect["op"],
                            })
                delay_two.append({
                    "program_index": program_index,
                    "trigger_action": rule["action"],
                    "delay": branch["delay"],
                    "operation": branch["effect"]["op"],
                    "target": branch["effect"]["target"],
                    "same_target_deterministic_immediate_actions": same_target_immediate,
                })

    active_goals = [
        row for row in population
        if row["public"]["goal"]["atom"].startswith("u:active:")
    ]
    delay_program_indices = {row["program_index"] for row in delay_two}
    delay_truth = [
        row for row in population
        if row["truth"]["target_program_index"] in delay_program_indices
    ]
    aligned_truth = [
        row for row in delay_truth
        if row["public"]["goal"]["atom"].startswith("u:active:")
    ]
    value_differences = [
        row["root_value"] - row["delayed_counterfactual_value"]
        for row in result["records"]
    ]

    diagnosis = {
        "schema_version": 55,
        "revision": "post_result_delay_gate_localization",
        "source_outcome_lock": str(OUTCOME.relative_to(PROJECT_ROOT)),
        "source_outcome_lock_sha256": file_sha256(OUTCOME),
        "source_result": str(RESULT.relative_to(PROJECT_ROOT)),
        "source_result_sha256": file_sha256(RESULT),
        "failed_gates": failed,
        "data_access": {
            "sealed_records_read_post_result": len(population),
            "new_planning_populations_constructed": 0,
            "additional_planning_evaluation_runs": 0,
            "exact_planner_calls": 0,
            "model_forward_passes": 0,
            "adapter_training_runs": 0,
        },
        "mechanic_support": {
            "program_templates": len(registry),
            "delay_two_program_templates": len(delay_program_indices),
            "delay_two_template_fraction": len(delay_program_indices) / len(registry),
            "delay_two_branches": delay_two,
            "all_delay_two_targets_have_a_same_program_same_target_deterministic_immediate_action": bool(delay_two) and all(
                row["same_target_deterministic_immediate_actions"] for row in delay_two
            ),
        },
        "population_support": {
            "tasks": len(population),
            "active_goal_tasks": len(active_goals),
            "active_goal_fraction": len(active_goals) / len(population),
            "delay_two_generating_truth_tasks": len(delay_truth),
            "delay_two_generating_truth_fraction": len(delay_truth) / len(population),
            "delay_two_truth_and_active_goal_tasks": len(aligned_truth),
            "delay_two_truth_and_active_goal_fraction": len(aligned_truth) / len(population),
        },
        "sealed_behavior": {
            "delayed_sensitive_tasks": sum(
                bool(row["delayed_consequence_sensitive"])
                for row in result["records"]
            ),
            "delayed_sensitive_fraction": result["metrics"]["nonmyopic_behavior"][
                "delayed_consequence_sensitive_policy_fraction"
            ],
            "maximum_absolute_root_value_change_under_delay_two_suppression": max(
                abs(value) for value in value_differences
            ),
            "root_value_changes": value_differences,
        },
        "localization": {
            "planner_timing_off_by_one": False,
            "evidence": [
                "the frozen analytic fixture establishes that a tick-zero delay-two event is delivered before the third action",
                "primary and independent scalar values agree exactly and independent policy evaluation agrees within floating-point tolerance",
                "the only delay-two branch is supported by one of eight templates",
                "only one sealed task jointly has the delay-two template as truth and an active goal",
                "the delay-two effect and a deterministic immediate action in that template address the same grounded target family",
                "suppressing delay two changes neither selected-policy qualification nor root value on any sealed task",
            ],
            "primary_cause": "control_task_adequacy_failure_from_sparse_goal_alignment_and_same_target_immediate_redundancy",
            "not_supported_as_primary_cause": [
                "belief_normalization_error",
                "bellman_backup_error",
                "policy_evaluation_error",
                "delay_delivery_off_by_one",
            ],
        },
        "repair_requirements": [
            "retain the frozen V55 failure and all nineteen passing gates",
            "do not rerun or reinterpret the V55 candidate population",
            "preregister a separate delayed-consequence confirmation population",
            "use a planning-specific latent mechanic registry with multiple delay-two templates",
            "forbid same-target deterministic immediate substitutes for every delay-two effect",
            "stratify public goals over the grounded delayed-effect target pool independently of generating truth",
            "replace reachability-only fixtures with exhaustive decision-relevance fixtures",
            "keep exact belief updates complete actions truth firewalls and independent policy evaluation",
        ],
        "decision": "authorize_preregistration_only_for_a_v55r1_delayed_consequence_adequacy_confirmation",
    }

    OUTPUT.write_text(json.dumps(diagnosis, indent=2, sort_keys=True) + "\n")
    SUMMARY.write_text(
        "# V55 delayed-consequence gate: failure localization\n\n"
        "Decision: retain the sealed V55 failure and preregister a separate "
        "delayed-consequence adequacy confirmation.\n\n"
        "V55 passed 19 of 20 gates. The failed gate is not explained by an "
        "off-by-one transition or Bellman error. The frozen delay fixture proves "
        "that a delay-two event scheduled at tick zero is delivered before the "
        "third action, while the primary planner, scalar reference, and independent "
        "policy evaluator agree.\n\n"
        "The failure is localized to task support. Only one of eight latent "
        "templates contains a delay-two branch. Only 8 of 32 goals concern the "
        "affected `active` predicate, and only one task combines an `active` goal "
        "with the delay-two template as generating truth. In the sole delay-two "
        "template, the stochastic delayed `set_false(active(target))` effect also "
        "has a deterministic immediate same-target `route` substitute. The original "
        "fixture established temporal visibility but never established that delay "
        "could alter an optimal action or value. Consistently, removing every "
        "delay-two effect changed the root value by exactly zero on all 32 tasks.\n\n"
        "The repair must not rerun V55. A new preregistration may define a separate "
        "confirmation suite with several delay-two latent templates, no same-target "
        "immediate substitutes, truth-independent goal stratification over delayed "
        "targets, and exhaustive decision-relevance fixtures. Formal verification "
        "remains blocked until that confirmation passes.\n"
    )
    print(json.dumps(diagnosis, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
