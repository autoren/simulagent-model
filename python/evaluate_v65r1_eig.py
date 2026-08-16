#!/usr/bin/env python3
"""One-shot V65r1 pooled SMC²-to-exact EIG portability evaluation."""
from __future__ import annotations

import argparse
import json
import math
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT
from v64_external_eig import (
    filter_public_history,
    identity_posterior,
    load_family,
    score_all_actions as exact_score_all_actions,
    select_action as exact_select_action,
    static_posterior,
)
from v65_smc2_eig import (
    attempted_outcome_leak,
    collapse_map_identity,
    collapse_theta_mean,
    force_equal_identity_evidence,
    pool_repeats,
    posterior_summary,
    rao_blackwellize_measure,
    score_all_actions,
    score_state_as_target,
    select_action,
    smc2_inference,
)


WORK_FIELDS = (
    "outer_particles_initialized",
    "inner_initial_draw_count",
    "inner_transition_draw_count",
    "observation_weight_evaluation_count",
    "complete_history_likelihood_recomputation_count",
    "inner_resampling_count",
    "outer_resampling_count",
    "pmmh_attempt_count",
    "pmmh_accept_count",
    "final_posterior_atom_count",
)


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def write_jsonl(path: Path, rows: Sequence[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in rows)
    )


def q95(values: Sequence[float]) -> float:
    return float(np.quantile(np.asarray(values, dtype=np.float64), 0.95))


def total_variation(left: Sequence[float], right: Sequence[float]) -> float:
    return 0.5 * float(
        np.sum(np.abs(np.asarray(left, dtype=np.float64) - np.asarray(right, dtype=np.float64)))
    )


def weighted_wasserstein_1(
    left_values: Sequence[float],
    left_weights: Sequence[float],
    right_values: Sequence[float],
    right_weights: Sequence[float],
) -> float:
    left_order = np.argsort(left_values)
    right_order = np.argsort(right_values)
    x = np.asarray(left_values, dtype=np.float64)[left_order]
    wx = np.asarray(left_weights, dtype=np.float64)[left_order]
    y = np.asarray(right_values, dtype=np.float64)[right_order]
    wy = np.asarray(right_weights, dtype=np.float64)[right_order]
    wx /= wx.sum()
    wy /= wy.sum()
    points = np.unique(np.concatenate([x, y]))
    if len(points) < 2:
        return 0.0
    left_cdf = np.searchsorted(x, points[:-1], side="right")
    right_cdf = np.searchsorted(y, points[:-1], side="right")
    left_mass = np.asarray([wx[:index].sum() for index in left_cdf])
    right_mass = np.asarray([wy[:index].sum() for index in right_cdf])
    return float(np.sum(np.abs(left_mass - right_mass) * np.diff(points)))


def _exact_summary(family, belief: np.ndarray, bins: int = 16) -> dict[str, Any]:
    static = static_posterior(belief)
    theta = static.sum(axis=0)
    joint = np.zeros((2, bins), dtype=np.float64)
    low, high = family.theta_support
    for identity in range(2):
        for node, value in enumerate(family.theta):
            index = min(bins - 1, max(0, int((value - low) / (high - low) * bins)))
            joint[identity, index] += float(static[identity, node])
    return {
        "identity": identity_posterior(belief),
        "theta_values": family.theta,
        "theta_weights": theta,
        "joint_bins": joint,
        "state": belief.sum(axis=(0, 1)),
    }


def exact_reference(family, record: dict[str, Any]) -> dict[str, Any]:
    belief, log_evidence = filter_public_history(
        family,
        record["initial_observation"],
        record["actions"],
        record["observations"],
    )
    scores = exact_score_all_actions(family, belief)
    selection = exact_select_action(family, belief)
    return {
        "belief": belief,
        "summary": _exact_summary(family, belief),
        "scores": scores,
        "values": np.asarray([row["eig"] for row in scores], dtype=np.float64),
        "predictives": np.asarray([row["predictive"] for row in scores], dtype=np.float64),
        "selected_action": selection["selected"]["action"],
        "optimal_actions": selection["optimal_actions"],
        "maximum": float(selection["maximum"]),
        "log_evidence": float(log_evidence),
    }


def _selection_from_values(actions: Sequence[str], values: Sequence[float]) -> str:
    maximum = max(float(value) for value in values)
    return next(
        action
        for action, value in zip(actions, values, strict=True)
        if float(value) >= maximum - 1e-12
    )


def _selection_metrics(
    exact: dict[str, Any], actions: Sequence[str], approximate_values: Sequence[float]
) -> dict[str, Any]:
    selected = _selection_from_values(actions, approximate_values)
    selected_index = actions.index(selected)
    regret = float(exact["maximum"] - exact["values"][selected_index])
    return {
        "selected_action": selected,
        "strict_optimal_membership": selected in exact["optimal_actions"],
        "epsilon_optimal_membership": regret <= 0.001 + 1e-15,
        "exact_regret": regret,
        "exact_optimal_actions": list(exact["optimal_actions"]),
    }


def compare_record_budget(
    family,
    record: dict[str, Any],
    exact: dict[str, Any],
    repeats: Sequence[dict[str, Any]],
    budget: int,
) -> dict[str, Any]:
    pooled = pool_repeats(repeats)
    repaired = rao_blackwellize_measure(family, pooled, record)
    approximate_scores = score_all_actions(family, repaired)
    approximate_values = np.asarray([row["eig"] for row in approximate_scores])
    approximate_predictives = np.asarray([row["predictive"] for row in approximate_scores])
    pooled_summary = posterior_summary(family, pooled)
    exact_summary = exact["summary"]
    actions = [row["action"] for row in approximate_scores]
    selection = _selection_metrics(exact, actions, approximate_values)
    implementation_selection = select_action(family, repaired)["selected"]["action"]
    repeat_rows = []
    repeat_selections = []
    for repeat in repeats:
        repeat_repaired = rao_blackwellize_measure(
            family, repeat, record, allow_unpooled_fixture=True
        )
        repeat_scores = score_all_actions(family, repeat_repaired)
        repeat_values = np.asarray([row["eig"] for row in repeat_scores])
        repeat_selection = _selection_metrics(exact, actions, repeat_values)
        repeat_selections.append(repeat_selection["selected_action"])
        repeat_rows.append(
            {
                "repeat": int(repeat["repeat"]),
                "values": repeat_values.tolist(),
                "mean_absolute_eig_error": float(np.mean(np.abs(repeat_values - exact["values"]))),
                **repeat_selection,
                "runtime_seconds": float(repeat["diagnostics"]["runtime_seconds"]),
                "work": repeat["diagnostics"]["work"],
                "random_stream_count": int(repeat["diagnostics"]["random_stream_count"]),
                "random_stream_collision_count": int(
                    repeat["diagnostics"]["random_stream_collision_count"]
                ),
                "normalizes": bool(repeat["normalizes"]),
            }
        )
    repeat_regrets = [row["exact_regret"] for row in repeat_rows]
    predictive_tvs = [
        total_variation(left, right)
        for left, right in zip(approximate_predictives, exact["predictives"], strict=True)
    ]
    controls: dict[str, dict[str, Any]] = {}
    average_repeat_values = np.mean([row["values"] for row in repeat_rows], axis=0)
    controls["average_repeat_EIG"] = {
        "values": average_repeat_values.tolist(),
        **_selection_metrics(exact, actions, average_repeat_values),
    }
    controls["first_repeat_only"] = {
        "values": repeat_rows[0]["values"],
        **_selection_metrics(exact, actions, repeat_rows[0]["values"]),
    }
    state_values = np.asarray(
        [score_state_as_target(family, pooled, action) for action in family.canonical_actions]
    )
    controls["state_as_target"] = {
        "values": state_values.tolist(),
        **_selection_metrics(exact, actions, state_values),
    }
    for name, controlled in (
        ("map_identity", collapse_map_identity(pooled)),
        ("theta_mean", collapse_theta_mean(pooled)),
        ("equal_identity_evidence", force_equal_identity_evidence(pooled)),
    ):
        controlled_rb = rao_blackwellize_measure(family, controlled, record)
        values = np.asarray([row["eig"] for row in score_all_actions(family, controlled_rb)])
        controls[name] = {"values": values.tolist(), **_selection_metrics(exact, actions, values)}
    plugin_values = np.asarray([row["eig"] for row in score_all_actions(family, pooled)])
    controls["plugin_particle_state_predictive"] = {
        "values": plugin_values.tolist(),
        **_selection_metrics(exact, actions, plugin_values),
    }
    for control in controls.values():
        control["mean_absolute_eig_error"] = float(
            np.mean(np.abs(np.asarray(control["values"]) - exact["values"]))
        )

    return {
        "record_id": record["record_id"],
        "prefix_length": int(record["prefix_length"]),
        "budget": int(budget),
        "pooled_normalizes": bool(pooled["normalizes"] and pooled_summary["normalizes"]),
        "candidate_predictive_normalizes": all(row["normalizes"] for row in approximate_scores),
        "candidate_count": len(actions),
        "candidate_order": actions,
        "tie_break_valid": selection["selected_action"] == implementation_selection,
        "finite": bool(
            all(row["finite"] for row in approximate_scores)
            and np.all(np.isfinite(approximate_values))
        ),
        "identity_tv": total_variation(pooled_summary["identity"], exact_summary["identity"]),
        "theta_wasserstein": weighted_wasserstein_1(
            pooled_summary["theta_values"],
            pooled_summary["theta_weights"],
            exact_summary["theta_values"],
            exact_summary["theta_weights"],
        ),
        "joint_identity_theta_tv": total_variation(
            pooled_summary["joint_bins"].ravel(), exact_summary["joint_bins"].ravel()
        ),
        "state_tv": total_variation(pooled_summary["state"], exact_summary["state"]),
        "candidate_predictive_tvs": predictive_tvs,
        "mean_candidate_predictive_tv": float(np.mean(predictive_tvs)),
        "approximate_values": approximate_values.tolist(),
        "exact_values": exact["values"].tolist(),
        "absolute_eig_errors": np.abs(approximate_values - exact["values"]).tolist(),
        "mean_absolute_eig_error": float(np.mean(np.abs(approximate_values - exact["values"]))),
        **selection,
        "repeat_diagnostics": repeat_rows,
        "repeat_selected_action_disagreement": len(set(repeat_selections)) > 1,
        "best_minus_worst_repeat_regret_spread": float(max(repeat_regrets) - min(repeat_regrets)),
        "controls": controls,
    }


def _metric_summary(rows: Sequence[dict], field: str) -> dict[str, float]:
    values = [float(row[field]) for row in rows]
    return {"mean": float(np.mean(values)), "q95": q95(values), "maximum": max(values)}


def _repeat_summary(rows: Sequence[dict], budget: int) -> dict[str, Any]:
    result = {}
    budget_rows = [row for row in rows if row["budget"] == budget]
    for repeat in range(3):
        diagnostics = [
            diagnostic
            for row in budget_rows
            for diagnostic in row["repeat_diagnostics"]
            if diagnostic.get("repeat") == repeat
        ]
        if len(diagnostics) != len(budget_rows):
            result[str(repeat)] = {
                "complete": False,
                "records": len(diagnostics),
                "mean_absolute_eig_error": math.inf,
                "strict_optimal_membership_rate": 0.0,
                "epsilon_optimal_membership_rate": 0.0,
                "mean_exact_regret": math.inf,
                "q95_exact_regret": math.inf,
                "selected_action_counts": {},
                "mean_runtime_seconds": math.inf,
            }
            continue
        result[str(repeat)] = {
            "complete": True,
            "records": len(diagnostics),
            "mean_absolute_eig_error": float(
                np.mean([row["mean_absolute_eig_error"] for row in diagnostics])
            ),
            "strict_optimal_membership_rate": float(
                np.mean([row["strict_optimal_membership"] for row in diagnostics])
            ),
            "epsilon_optimal_membership_rate": float(
                np.mean([row["epsilon_optimal_membership"] for row in diagnostics])
            ),
            "mean_exact_regret": float(np.mean([row["exact_regret"] for row in diagnostics])),
            "q95_exact_regret": q95([row["exact_regret"] for row in diagnostics]),
            "selected_action_counts": dict(Counter(row["selected_action"] for row in diagnostics)),
            "mean_runtime_seconds": float(
                np.mean([row["runtime_seconds"] for row in diagnostics])
            ),
        }
    return result


def _compute_summary(rows: Sequence[dict], budgets: Sequence[int]) -> dict[str, Any]:
    result: dict[str, Any] = {"by_budget": {}, "by_prefix_length": {}, "cells": []}
    for row in rows:
        for repeat in row["repeat_diagnostics"]:
            cell = {
                "record_id": row["record_id"],
                "prefix_length": row["prefix_length"],
                "budget": row["budget"],
                "repeat": repeat["repeat"],
                "runtime_seconds": repeat["runtime_seconds"],
                "random_stream_count": repeat["random_stream_count"],
                "random_stream_collision_count": repeat["random_stream_collision_count"],
                **{key: int(repeat["work"][key]) for key in WORK_FIELDS},
            }
            result["cells"].append(cell)
    for budget in budgets:
        cells = [row for row in result["cells"] if row["budget"] == budget]
        result["by_budget"][str(budget)] = {
            "cells": len(cells),
            "runtime_seconds_total": float(sum(row["runtime_seconds"] for row in cells)),
            "runtime_seconds_mean": float(np.mean([row["runtime_seconds"] for row in cells])),
            "runtime_seconds_q95": q95([row["runtime_seconds"] for row in cells]),
            "work_totals": {key: sum(row[key] for row in cells) for key in WORK_FIELDS},
        }
    for prefix in range(6):
        cells = [row for row in result["cells"] if row["prefix_length"] == prefix]
        result["by_prefix_length"][str(prefix)] = {
            "cells": len(cells),
            "runtime_seconds_total": float(sum(row["runtime_seconds"] for row in cells)),
            "runtime_seconds_mean": float(np.mean([row["runtime_seconds"] for row in cells])),
        }
    return result


def aggregate_evaluation(
    rows: Sequence[dict],
    config: dict[str, Any],
    implementation_audit: dict[str, Any],
    access: dict[str, int],
) -> dict[str, Any]:
    budgets = config["smcSquared"]["outerThetaParticleBudgets"]
    primary_budget = config["smcSquared"]["primaryOuterThetaParticleBudget"]
    low_budget = min(budgets)
    expected_rows = config["subset"]["records"] * len(budgets)
    expected_cells = expected_rows * config["smcSquared"]["independentRepeatsPerBudget"]
    expected_repeats = config["smcSquared"]["independentRepeatsPerBudget"]
    unique_record_ids = {row["record_id"] for row in rows}
    unique_record_budget = {(row["record_id"], row["budget"]) for row in rows}
    complete_grid = bool(
        len(rows) == expected_rows
        and len(unique_record_ids) == config["subset"]["records"]
        and len(unique_record_budget) == expected_rows
        and all(
            [repeat["repeat"] for repeat in row["repeat_diagnostics"]]
            == list(range(expected_repeats))
            for row in rows
        )
        and all(
            {row["budget"] for row in rows if row["record_id"] == record_id}
            == set(budgets)
            for record_id in unique_record_ids
        )
    )
    primary = [row for row in rows if row["budget"] == primary_budget]
    low = [row for row in rows if row["budget"] == low_budget]
    by_budget = {}
    for budget in budgets:
        budget_rows = [row for row in rows if row["budget"] == budget]
        action_errors = [value for row in budget_rows for value in row["absolute_eig_errors"]]
        predictive_tvs = [value for row in budget_rows for value in row["candidate_predictive_tvs"]]
        by_budget[str(budget)] = {
            "records": len(budget_rows),
            "identity_tv": _metric_summary(budget_rows, "identity_tv"),
            "theta_wasserstein": _metric_summary(budget_rows, "theta_wasserstein"),
            "joint_identity_theta_tv": _metric_summary(budget_rows, "joint_identity_theta_tv"),
            "state_tv": _metric_summary(budget_rows, "state_tv"),
            "candidate_predictive_tv": {
                "mean": float(np.mean(predictive_tvs)),
                "q95": q95(predictive_tvs),
                "maximum": max(predictive_tvs),
            },
            "absolute_eig_vector_error": {
                "mean": float(np.mean(action_errors)),
                "q95": q95(action_errors),
                "maximum": max(action_errors),
            },
            "strict_optimal_membership_rate": float(
                np.mean([row["strict_optimal_membership"] for row in budget_rows])
            ),
            "epsilon_optimal_membership_rate": float(
                np.mean([row["epsilon_optimal_membership"] for row in budget_rows])
            ),
            "selected_eig_regret": _metric_summary(budget_rows, "exact_regret"),
            "selected_action_counts": dict(Counter(row["selected_action"] for row in budget_rows)),
            "repeat_selected_action_disagreement_rate": float(
                np.mean([row["repeat_selected_action_disagreement"] for row in budget_rows])
            ),
            "repeat_regret_spread": _metric_summary(
                budget_rows, "best_minus_worst_repeat_regret_spread"
            ),
            "single_repeat": _repeat_summary(rows, budget),
        }

    primary_mean_error = by_budget[str(primary_budget)]["absolute_eig_vector_error"]["mean"]
    control_names = list(primary[0]["controls"]) if primary else []
    controls = {}
    for name in control_names:
        controls_rows = [row["controls"][name] for row in primary]
        mean_regret = float(np.mean([row["exact_regret"] for row in controls_rows]))
        disagreement = float(
            np.mean([not row["strict_optimal_membership"] for row in controls_rows])
        )
        mean_error = float(np.mean([row["mean_absolute_eig_error"] for row in controls_rows]))
        controls[name] = {
            "mean_exact_regret": mean_regret,
            "strict_selection_disagreement_rate": disagreement,
            "mean_absolute_eig_error": mean_error,
            "detected_or_dominated": bool(
                mean_regret > 0.001
                or disagreement > 0.10
                or mean_error > primary_mean_error + 0.001
            ),
        }
    controls["shared_stream"] = {
        "detected_or_dominated": bool(
            implementation_audit["mutation_audit"]["checks"][
                "share_inner_streams_across_outer_particles"
            ]
        ),
        "source": "frozen_implementation_mutation_audit",
    }
    leak_rejected = False
    if primary:
        try:
            attempted_outcome_leak(
                {
                    "record_id": primary[0]["record_id"],
                    "prefix_length": primary[0]["prefix_length"],
                    "initial_observation": "left",
                    "actions": [],
                    "observations": [],
                },
                "future",
            )
        except PermissionError:
            leak_rejected = True
    controls["outcome_leakage"] = {
        "detected_or_dominated": leak_rejected,
        "firewall_rejected": leak_rejected,
    }
    detected = sum(row["detected_or_dominated"] for row in controls.values())
    compute = _compute_summary(rows, budgets)
    work_complete = bool(
        len(compute["cells"]) == expected_cells
        and all(
            all(field in cell and cell[field] >= 0 for field in WORK_FIELDS)
            for cell in compute["cells"]
        )
    )
    gates = config["gates"]
    primary_summary = by_budget[str(primary_budget)]
    low_summary = by_budget[str(low_budget)]
    checks = {
        "completed_record_budget_repeat_fraction": (
            complete_grid and len(compute["cells"]) == expected_cells
        ),
        "pooled_posterior_normalization": all(row["pooled_normalizes"] for row in rows),
        "candidate_and_predictive_normalization": all(
            row["candidate_predictive_normalizes"] for row in rows
        ),
        "finite_values": all(row["finite"] for row in rows),
        "primary_mean_identity_tv": primary_summary["identity_tv"]["mean"]
        <= gates["maximumPrimaryMeanIdentityTv"],
        "primary_q95_identity_tv": primary_summary["identity_tv"]["q95"]
        <= gates["maximumPrimaryQ95IdentityTv"],
        "primary_mean_theta_wasserstein": primary_summary["theta_wasserstein"]["mean"]
        <= gates["maximumPrimaryMeanThetaWasserstein"],
        "primary_q95_theta_wasserstein": primary_summary["theta_wasserstein"]["q95"]
        <= gates["maximumPrimaryQ95ThetaWasserstein"],
        "primary_mean_joint_identity_theta_tv": primary_summary[
            "joint_identity_theta_tv"
        ]["mean"]
        <= gates["maximumPrimaryMeanJointIdentityThetaTv"],
        "primary_q95_joint_identity_theta_tv": primary_summary[
            "joint_identity_theta_tv"
        ]["q95"]
        <= gates["maximumPrimaryQ95JointIdentityThetaTv"],
        "primary_mean_state_tv": primary_summary["state_tv"]["mean"]
        <= gates["maximumPrimaryMeanStateTv"],
        "primary_q95_state_tv": primary_summary["state_tv"]["q95"]
        <= gates["maximumPrimaryQ95StateTv"],
        "primary_mean_candidate_predictive_tv": primary_summary[
            "candidate_predictive_tv"
        ]["mean"]
        <= gates["maximumPrimaryMeanCandidatePredictiveTv"],
        "primary_q95_candidate_predictive_tv": primary_summary[
            "candidate_predictive_tv"
        ]["q95"]
        <= gates["maximumPrimaryQ95CandidatePredictiveTv"],
        "primary_mean_EIG_vector_error": primary_summary["absolute_eig_vector_error"][
            "mean"
        ]
        <= gates["maximumPrimaryMeanAbsoluteEigVectorErrorNats"],
        "primary_q95_EIG_vector_error": primary_summary["absolute_eig_vector_error"][
            "q95"
        ]
        <= gates["maximumPrimaryQ95AbsoluteEigVectorErrorNats"],
        "primary_strict_optimal_membership": primary_summary[
            "strict_optimal_membership_rate"
        ]
        >= gates["minimumPrimaryStrictOptimalSetMembershipRate"],
        "primary_epsilon_optimal_membership": primary_summary[
            "epsilon_optimal_membership_rate"
        ]
        >= gates["minimumPrimaryEpsilonOptimalMembershipRate"],
        "primary_mean_selected_regret": primary_summary["selected_eig_regret"]["mean"]
        <= gates["maximumPrimaryMeanSelectedEigRegretNats"],
        "primary_q95_selected_regret": primary_summary["selected_eig_regret"]["q95"]
        <= gates["maximumPrimaryQ95SelectedEigRegretNats"],
        "primary_maximum_selected_regret": primary_summary["selected_eig_regret"]["maximum"]
        <= gates["maximumPrimarySelectedEigRegretNats"],
        "primary_minus_low_mean_EIG_error": (
            primary_summary["absolute_eig_vector_error"]["mean"]
            - low_summary["absolute_eig_vector_error"]["mean"]
        )
        <= gates["maximumPrimaryMinusLowMeanEigVectorErrorNats"],
        "primary_minus_low_mean_selected_regret": (
            primary_summary["selected_eig_regret"]["mean"]
            - low_summary["selected_eig_regret"]["mean"]
        )
        <= gates["maximumPrimaryMinusLowMeanSelectedEigRegretNats"],
        "controls_detected_or_dominated": detected
        >= gates["minimumControlsDetectedOrDominated"],
        "implementation_mutant_kill_rate": implementation_audit["mutation_audit"][
            "kill_rate"
        ]
        >= gates["minimumImplementationMutantKillRate"],
        "analytic_fixture_pass_rate": implementation_audit["analytic_fixtures"][
            "pass_rate"
        ]
        >= gates["minimumAnalyticFixturePassRate"],
        "complete_compute_diagnostics": work_complete,
        "access_and_one_shot_integrity": bool(
            access["logical_evaluation_attempts"] == 1
            and access["subset_public_records_loaded"] == config["subset"]["records"]
            and access["v64_source_public_records_loaded_during_evaluation"] == 0
            and access["v64_selection_audit_records_loaded"] == 0
            and access["v64_evaluation_records_loaded"] == 0
            and access["truth_field_access_count"] == 0
            and access["realized_outcome_access_before_selection_count"] == 0
            and access["candidate_omission_count"] == 0
            and access["tie_break_violation_count"] == 0
            and access["random_stream_collision_count"] == 0
            and access["human_record_access_count"] == 0
            and access["model_forward_pass_count"] == 0
            and access["adapter_training_run_count"] == 0
        ),
    }
    failed = [name for name, passed in checks.items() if not passed]
    return {
        "schema_version": "65r1",
        "experiment": "v65r1_pooled_smc2_eig_portability",
        "passed": not failed,
        "decision": (
            "authorize_preregistration_of_external_Bayes_adaptive_reward_decisions"
            if not failed
            else "do_not_authorize_reward_planning"
        ),
        "failed_gates": failed,
        "gate_checks": checks,
        "by_budget": by_budget,
        "controls": {
            "controls": controls,
            "detected_or_dominated": detected,
            "rule": "mean_exact_regret_gt_0.001_or_strict_disagreement_gt_0.10_or_mean_EIG_error_gt_primary_plus_0.001;_shared_stream_and_outcome_leakage_require_direct_detection",
        },
        "compute_diagnostics": compute,
        "access": access,
        "claim_boundary": {
            "Rao_Blackwellized_known_state_for_acquisition": True,
            "SMC2_static_identity_theta_posterior": True,
            "particle_state_posterior_separately_evaluated": True,
            "pure_nested_particle_predictive_qualified": False,
            "paired_frozen_V64_history_reuse": True,
            "independent_exact_replication": False,
            "sequential_approximate_adaptive_rollout_tested": False,
            "reward_planning_tested": False,
            "formal_verification_tested": False,
            "human_or_model_access": False,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lock", default="configs/v65r1-evaluation-implementation-lock.json")
    parser.add_argument(
        "--output", default="outputs/v65r1-nested-predictive-repair/evaluation/result.json"
    )
    args = parser.parse_args()
    lock_path = (PROJECT_ROOT / args.lock).resolve()
    output_path = (PROJECT_ROOT / args.output).resolve()
    if output_path.exists():
        raise RuntimeError("V65r1 immutable evaluation result already exists")
    lock = json.loads(lock_path.read_text())
    if not lock["authorization"]["run_one_immutable_evaluation"]:
        raise RuntimeError("V65r1 evaluator lock does not authorize evaluation")
    for relative, digest in lock["source_sha256"].items():
        if file_sha256(PROJECT_ROOT / relative) != digest:
            raise RuntimeError(f"frozen V65r1 evaluator or dependency changed: {relative}")
    subset_seal_path = (PROJECT_ROOT / lock["subset_seal"]).resolve()
    if file_sha256(subset_seal_path) != lock["subset_seal_sha256"]:
        raise RuntimeError("V65r1 subset seal changed after evaluator freeze")
    subset_seal = json.loads(subset_seal_path.read_text())
    implementation_path = (PROJECT_ROOT / subset_seal["implementation_lock"]).resolve()
    implementation = json.loads(implementation_path.read_text())
    design_path = (PROJECT_ROOT / implementation["design_lock"]).resolve()
    design = json.loads(design_path.read_text())
    config = design["config_payload"]
    implementation_audit = json.loads(
        (PROJECT_ROOT / implementation["implementation_audit"]).read_text()
    )
    subset_path = (PROJECT_ROOT / subset_seal["files"]["subset_public"]["path"]).resolve()
    if file_sha256(subset_path) != subset_seal["files"]["subset_public"]["sha256"]:
        raise RuntimeError("V65r1 public subset changed after seal")
    records = read_jsonl(subset_path)
    family = load_family()
    started = time.perf_counter()
    rows = []
    stream_collisions = 0
    for record in records:
        exact = exact_reference(family, record)
        for budget in config["smcSquared"]["outerThetaParticleBudgets"]:
            repeats = [
                smc2_inference(family, record, config, int(budget), repeat)
                for repeat in range(config["smcSquared"]["independentRepeatsPerBudget"])
            ]
            stream_collisions += sum(
                row["diagnostics"]["random_stream_collision_count"] for row in repeats
            )
            rows.append(compare_record_budget(family, record, exact, repeats, int(budget)))
    access = {
        "logical_evaluation_attempts": 1,
        "subset_public_records_loaded": len(records),
        "v64_source_public_records_loaded_during_evaluation": 0,
        "v64_selection_audit_records_loaded": 0,
        "v64_evaluation_records_loaded": 0,
        "truth_field_access_count": 0,
        "realized_outcome_access_before_selection_count": 0,
        "candidate_omission_count": int(
            sum(
                row["candidate_count"] != len(config["approximateAcquisition"]["candidateOrder"])
                or row["candidate_order"] != config["approximateAcquisition"]["candidateOrder"]
                for row in rows
            )
        ),
        "tie_break_violation_count": int(sum(not row["tie_break_valid"] for row in rows)),
        "random_stream_collision_count": int(stream_collisions),
        "human_record_access_count": 0,
        "model_forward_pass_count": 0,
        "adapter_training_run_count": 0,
    }
    result = aggregate_evaluation(rows, config, implementation_audit, access)
    result["runtime_seconds"] = time.perf_counter() - started
    result["bindings"] = {
        "evaluation_implementation_lock": str(lock_path.relative_to(PROJECT_ROOT)),
        "evaluation_implementation_lock_sha256": file_sha256(lock_path),
        "subset_seal": str(subset_seal_path.relative_to(PROJECT_ROOT)),
        "subset_seal_sha256": file_sha256(subset_seal_path),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    raw_path = output_path.parent / "record-budget-cells.jsonl"
    write_jsonl(raw_path, rows)
    result["record_budget_cells"] = str(raw_path.relative_to(PROJECT_ROOT))
    result["record_budget_cells_sha256"] = file_sha256(raw_path)
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "passed": result["passed"],
        "decision": result["decision"],
        "failed_gates": result["failed_gates"],
        "by_budget": result["by_budget"],
        "controls": result["controls"],
        "runtime_seconds": result["runtime_seconds"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
