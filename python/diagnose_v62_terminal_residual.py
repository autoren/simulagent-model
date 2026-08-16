#!/usr/bin/env python3
"""Localize V62's Bellman residual failure without changing its result."""
from __future__ import annotations

import argparse
import json

import numpy as np

from evaluate_v62_external import load_model
from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT
from v62_external_pomdp import (
    ExactPlanner,
    all_positive_observation_beliefs,
    bellman_residual,
    terminal_mask,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outcome-lock", default="configs/v62-outcome-lock.json")
    parser.add_argument(
        "--output",
        default="outputs/v62-external-pomdp-transfer/post-hoc-terminal-residual-diagnostic.json",
    )
    parser.add_argument(
        "--summary", default="docs/v62-post-hoc-terminal-residual-diagnostic.md"
    )
    args = parser.parse_args()
    outcome_path, output, summary_path = tuple(
        (PROJECT_ROOT / value).resolve()
        for value in (args.outcome_lock, args.output, args.summary)
    )
    if output.exists() or summary_path.exists():
        raise RuntimeError("V62 post-hoc terminal diagnostic already exists")
    outcome = json.loads(outcome_path.read_text())
    result_path = PROJECT_ROOT / outcome["result"]
    result = json.loads(result_path.read_text())
    seal_path = PROJECT_ROOT / outcome["external_bundle_seal"]
    seal = json.loads(seal_path.read_text())
    bundle = PROJECT_ROOT / seal["bundle"]
    implementation = json.loads((PROJECT_ROOT / seal["implementation_lock"]).read_text())
    design = json.loads((PROJECT_ROOT / implementation["design_lock"]).read_text())
    config = design["config_payload"]
    threshold = config["gates"]["maximumIndependentBellmanResidual"]

    source_runtime_path = bundle / "source/pobax/envs/classic/pomdp.py"
    official_rollout_path = PROJECT_ROOT / "python/official_v62_rollout.py"
    runtime_text = source_runtime_path.read_text()
    rollout_text = official_rollout_path.read_text()
    official_terminal_semantics = (
        "terminal = is_absorbing.all()" in runtime_text
        and "continue_active = active & ~done" in rollout_text
        and "returns += active" in rollout_text
    )

    cells = []
    total_nodes = 0
    terminal_nodes = 0
    failing_nodes = 0
    nonterminal_failing_nodes = 0
    terminal_decision_violations = 0
    for declared in config["benchmark"]["models"]:
        model_id = declared["id"]
        model = load_model(bundle / f"models/{model_id}/model.json")
        terminals = terminal_mask(model)
        for horizon in declared["horizons"]:
            planner = ExactPlanner(model)
            planner.initial_value(horizon)
            rows = []
            for belief, remaining in all_positive_observation_beliefs(
                model, planner, horizon
            ):
                support = np.flatnonzero(belief > 1e-14)
                terminal_support = bool(np.all(terminals[support]))
                decision = planner.decision(belief, remaining)
                residual = bellman_residual(model, planner, belief, remaining)
                failure = residual > threshold
                total_nodes += 1
                terminal_nodes += int(terminal_support)
                failing_nodes += int(failure)
                nonterminal_failing_nodes += int(failure and not terminal_support)
                terminal_decision_violations += int(
                    terminal_support
                    and (
                        abs(decision.value) > threshold
                        or max(abs(value) for value in decision.q_values) > threshold
                    )
                )
                if failure:
                    rows.append(
                        {
                            "remaining_horizon": remaining,
                            "terminal_support": terminal_support,
                            "decision_value": decision.value,
                            "decision_q_values": list(decision.q_values),
                            "old_residual": residual,
                        }
                    )
            recorded = next(
                row
                for row in result["exact_records"]
                if row["model_id"] == model_id and row["horizon"] == horizon
            )
            cells.append(
                {
                    "model_id": model_id,
                    "horizon": horizon,
                    "recomputed_maximum_old_residual": max(
                        (row["old_residual"] for row in rows), default=0.0
                    )
                    if rows
                    else recorded["maximum_bellman_residual"],
                    "recorded_maximum_old_residual": recorded[
                        "maximum_bellman_residual"
                    ],
                    "failing_nodes": rows,
                }
            )

    # For passing cells, the maximum was not retained above; independently recompute all maxima.
    for cell in cells:
        model = load_model(bundle / f"models/{cell['model_id']}/model.json")
        planner = ExactPlanner(model)
        planner.initial_value(cell["horizon"])
        cell["recomputed_maximum_old_residual"] = max(
            (
                bellman_residual(model, planner, belief, remaining)
                for belief, remaining in all_positive_observation_beliefs(
                    model, planner, cell["horizon"]
                )
            ),
            default=0.0,
        )

    checks = {
        "v62_is_immutable_failed_outcome": (
            not outcome["qualification_passed"]
            and outcome["failed_checks"] == ["independent_bellman_residual"]
            and file_sha256(result_path) == outcome["result_sha256"]
        ),
        "old_residuals_reproduce_exactly": all(
            cell["recomputed_maximum_old_residual"]
            == cell["recorded_maximum_old_residual"]
            for cell in cells
        ),
        "every_old_residual_failure_is_terminal_only": (
            failing_nodes > 0 and nonterminal_failing_nodes == 0
        ),
        "planner_terminal_values_are_zero": terminal_decision_violations == 0,
        "official_runtime_stops_after_absorbing_successor": official_terminal_semantics,
        "no_new_rollout_or_candidate_evaluation": (
            json.loads((PROJECT_ROOT / outcome["evaluation_attempt"]).read_text())[
                "attempt"
            ]
            == 1
        ),
    }
    diagnostic = {
        "schema_version": "62-post-hoc",
        "experiment": "v62_terminal_residual_post_hoc_diagnostic",
        "passed": all(checks.values()),
        "status": "diagnostic_only_does_not_change_v62_qualification",
        "outcome_lock": str(outcome_path.relative_to(PROJECT_ROOT)),
        "outcome_lock_sha256": file_sha256(outcome_path),
        "result": str(result_path.relative_to(PROJECT_ROOT)),
        "result_sha256": file_sha256(result_path),
        "official_runtime_source": str(source_runtime_path.relative_to(PROJECT_ROOT)),
        "official_runtime_source_sha256": file_sha256(source_runtime_path),
        "official_rollout": str(official_rollout_path.relative_to(PROJECT_ROOT)),
        "official_rollout_sha256": file_sha256(official_rollout_path),
        "node_counts": {
            "reachable": total_nodes,
            "terminal_support": terminal_nodes,
            "old_residual_failures": failing_nodes,
            "nonterminal_old_residual_failures": nonterminal_failing_nodes,
            "terminal_decision_violations": terminal_decision_violations,
        },
        "cells": cells,
        "checks": checks,
        "data_access": {
            "new_external_rollouts": 0,
            "new_candidate_evaluations": 0,
            "human_records": 0,
            "model_forward_passes": 0,
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(diagnostic, indent=2, sort_keys=True) + "\n")
    summary = f"""# V62 post-hoc terminal-residual diagnostic

This is a labeled post-hoc diagnostic. It does not revise V62's immutable failed qualification.

- All `{failing_nodes}` reachable nodes whose old residual exceeded `1e-10` had belief support entirely on absorbing terminal states.
- There were `{nonterminal_failing_nodes}` nonterminal residual failures.
- The planner returned zero value and zero action values at every terminal-support node, with `{terminal_decision_violations}` violations.
- The pinned POBAX runtime declares an all-action absorbing successor terminal, and the frozen rollout excludes done episodes before the next reward-bearing step.
- The original per-cell residuals reproduced exactly without another rollout or candidate evaluation.

The old residual checker special-cased horizon zero but not terminal belief support. It therefore recomposed a counterfactual action reward after the runtime and planner had already stopped. The proper next step is a separately preregistered measurement repair that adds the missing terminal-support base case, mutation-tests that rule, and rescores only the immutable reachable belief nodes. V62 itself remains a 31/32-gate failure.
"""
    summary_path.write_text(summary)
    print(json.dumps(diagnostic, indent=2, sort_keys=True))
    if not diagnostic["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
