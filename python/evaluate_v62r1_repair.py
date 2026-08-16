#!/usr/bin/env python3
"""Run the one authorized V62r1 terminal-residual repair rescore."""
from __future__ import annotations

import argparse
import hashlib
import json

import numpy as np

from evaluate_v62_external import load_model
from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT
from v62_external_pomdp import (
    ExactPlanner,
    all_positive_observation_beliefs,
    bellman_residual,
)
from v62r1_terminal_residual import (
    support_is_all_action_absorbing,
    terminal_aware_bellman_residual,
)


def canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def qualify(metrics: dict[str, float], gates: dict[str, float]) -> dict[str, object]:
    checks = {
        "source_binding_rate": metrics["source_binding_rate"]
        >= gates["minimumSourceBindingRate"],
        "source_old_residual_reproduction_rate": metrics[
            "source_old_residual_reproduction_rate"
        ]
        >= gates["minimumSourceOldResidualReproductionRate"],
        "old_failure_terminal_localization_rate": metrics[
            "old_failure_terminal_localization_rate"
        ]
        >= gates["minimumOldFailureTerminalLocalizationRate"],
        "corrected_residual": metrics["maximum_corrected_residual"]
        <= gates["maximumCorrectedResidual"],
        "corrected_terminal_residual": metrics[
            "maximum_corrected_terminal_residual"
        ]
        <= gates["maximumCorrectedTerminalResidual"],
        "corrected_nonterminal_residual": metrics[
            "maximum_corrected_nonterminal_residual"
        ]
        <= gates["maximumCorrectedNonterminalResidual"],
        "other_v62_gate_reproduction_rate": metrics[
            "other_v62_gate_reproduction_rate"
        ]
        >= gates["minimumOtherV62GateReproductionRate"],
        "exact_record_delta": metrics["maximum_exact_record_delta"]
        <= gates["maximumExactRecordDelta"],
        "official_rollout_record_delta": metrics[
            "maximum_official_rollout_record_delta"
        ]
        <= gates["maximumOfficialRolloutRecordDelta"],
        "repair_fixture_pass_rate": metrics["repair_fixture_pass_rate"]
        >= gates["minimumRepairFixturePassRate"],
        "repair_mutant_kill_rate": metrics["repair_mutant_kill_rate"]
        >= gates["minimumRepairMutantKillRate"],
        "repair_attempt_and_access_violation_count": metrics[
            "repair_attempt_and_access_violation_count"
        ]
        <= gates["maximumRepairAttemptAndAccessViolationCount"],
    }
    return {"passed": all(checks.values()), "checks": checks}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--evaluation-lock", default="configs/v62r1-evaluation-implementation-lock.json"
    )
    parser.add_argument(
        "--attempt", default="outputs/v62r1-terminal-residual-repair/rescore-attempt.json"
    )
    parser.add_argument(
        "--output", default="outputs/v62r1-terminal-residual-repair/rescore/result.json"
    )
    args = parser.parse_args()
    evaluation_lock_path, attempt_path, output = tuple(
        (PROJECT_ROOT / value).resolve()
        for value in (args.evaluation_lock, args.attempt, args.output)
    )
    if attempt_path.exists() or output.exists():
        raise RuntimeError("V62r1 repair rescore already attempted")
    evaluation_lock = json.loads(evaluation_lock_path.read_text())
    implementation_lock_path = PROJECT_ROOT / evaluation_lock["implementation_lock"]
    implementation_lock = json.loads(implementation_lock_path.read_text())
    design_lock_path = PROJECT_ROOT / implementation_lock["design_lock"]
    design = json.loads(design_lock_path.read_text())
    config = design["config_payload"]
    outcome_path = PROJECT_ROOT / implementation_lock["source_v62_outcome_lock"]
    outcome = json.loads(outcome_path.read_text())
    source_result_path = PROJECT_ROOT / outcome["result"]
    source_result = json.loads(source_result_path.read_text())
    seal_path = PROJECT_ROOT / implementation_lock["source_v62_external_bundle_seal"]
    seal = json.loads(seal_path.read_text())
    bundle = PROJECT_ROOT / seal["bundle"]

    attempt = {
        "schema_version": "62r1",
        "experiment": "v62r1_repair_rescore_attempt",
        "attempt": 1,
        "evaluation_implementation_lock": str(
            evaluation_lock_path.relative_to(PROJECT_ROOT)
        ),
        "evaluation_implementation_lock_sha256": file_sha256(evaluation_lock_path),
        "source_v62_result": str(source_result_path.relative_to(PROJECT_ROOT)),
        "source_v62_result_sha256": file_sha256(source_result_path),
        "new_candidate_evaluations": 0,
        "new_external_rollouts": 0,
        "human_records": 0,
        "model_forward_passes": 0,
    }
    attempt_path.parent.mkdir(parents=True, exist_ok=True)
    attempt_path.write_text(json.dumps(attempt, indent=2, sort_keys=True) + "\n")

    source_record_by_cell = {
        (row["model_id"], row["horizon"]): row
        for row in source_result["exact_records"]
    }
    records = []
    all_corrected: list[float] = []
    terminal_corrected: list[float] = []
    nonterminal_corrected: list[float] = []
    old_failures = 0
    old_terminal_failures = 0
    reachable_nodes = 0
    if config["repairRescore"]["taskCells"] != 6:
        raise RuntimeError("frozen V62r1 task census changed")
    source_design = json.loads(
        (
            PROJECT_ROOT
            / json.loads((PROJECT_ROOT / seal["implementation_lock"]).read_text())[
                "design_lock"
            ]
        ).read_text()
    )
    for declared in source_design["config_payload"]["benchmark"]["models"]:
        model_id = declared["id"]
        model = load_model(bundle / f"models/{model_id}/model.json")
        for horizon in declared["horizons"]:
            planner = ExactPlanner(model)
            planner.initial_value(horizon)
            node_rows = []
            for belief, remaining in all_positive_observation_beliefs(
                model, planner, horizon
            ):
                old = bellman_residual(model, planner, belief, remaining)
                corrected = terminal_aware_bellman_residual(
                    model, planner, belief, remaining
                )
                terminal = support_is_all_action_absorbing(model, belief)
                failure = old > source_design["config_payload"]["gates"][
                    "maximumIndependentBellmanResidual"
                ]
                reachable_nodes += 1
                old_failures += int(failure)
                old_terminal_failures += int(failure and terminal)
                all_corrected.append(corrected)
                (terminal_corrected if terminal else nonterminal_corrected).append(corrected)
                node_rows.append(
                    {
                        "remaining_horizon": remaining,
                        "terminal_support": terminal,
                        "old_residual": old,
                        "corrected_residual": corrected,
                    }
                )
            source_record = source_record_by_cell[(model_id, horizon)]
            records.append(
                {
                    "model_id": model_id,
                    "horizon": horizon,
                    "reachable_belief_count": len(node_rows),
                    "recorded_old_maximum_residual": source_record[
                        "maximum_bellman_residual"
                    ],
                    "recomputed_old_maximum_residual": max(
                        (row["old_residual"] for row in node_rows), default=0.0
                    ),
                    "corrected_maximum_residual": max(
                        (row["corrected_residual"] for row in node_rows), default=0.0
                    ),
                    "terminal_support_nodes": sum(
                        row["terminal_support"] for row in node_rows
                    ),
                    "old_residual_failing_nodes": sum(
                        row["old_residual"]
                        > source_design["config_payload"]["gates"][
                            "maximumIndependentBellmanResidual"
                        ]
                        for row in node_rows
                    ),
                    "nodes": node_rows,
                }
            )

    copied_exact_records = json.loads(json.dumps(source_result["exact_records"]))
    copied_rollout_records = json.loads(json.dumps(source_result["official_rollout_records"]))
    other_v62_checks = {
        key: value
        for key, value in outcome["gate_checks"].items()
        if key != "independent_bellman_residual"
    }
    source_binding_checks = {
        "evaluation_lock": file_sha256(evaluation_lock_path)
        == attempt["evaluation_implementation_lock_sha256"],
        "implementation_lock": file_sha256(implementation_lock_path)
        == evaluation_lock["implementation_lock_sha256"],
        "design_lock": file_sha256(design_lock_path)
        == implementation_lock["design_lock_sha256"],
        "outcome_lock": file_sha256(outcome_path)
        == implementation_lock["source_v62_outcome_lock_sha256"],
        "source_result": file_sha256(source_result_path) == outcome["result_sha256"],
        "external_bundle_seal": file_sha256(seal_path)
        == implementation_lock["source_v62_external_bundle_seal_sha256"],
        "repair_implementation": file_sha256(
            PROJECT_ROOT / implementation_lock["implementation"]
        )
        == implementation_lock["implementation_sha256"],
    }
    old_reproductions = [
        row["recomputed_old_maximum_residual"]
        == row["recorded_old_maximum_residual"]
        for row in records
    ]
    access_violations = sum(
        (
            int(attempt["attempt"] != 1),
            int(json.loads((PROJECT_ROOT / outcome["evaluation_attempt"]).read_text())["attempt"] != 1),
            attempt["new_candidate_evaluations"],
            attempt["new_external_rollouts"],
            attempt["human_records"],
            attempt["model_forward_passes"],
        )
    )
    metrics = {
        "source_binding_rate": sum(source_binding_checks.values())
        / len(source_binding_checks),
        "source_old_residual_reproduction_rate": sum(old_reproductions)
        / len(old_reproductions),
        "old_failure_terminal_localization_rate": old_terminal_failures / old_failures,
        "maximum_corrected_residual": max(all_corrected, default=0.0),
        "maximum_corrected_terminal_residual": max(terminal_corrected, default=0.0),
        "maximum_corrected_nonterminal_residual": max(
            nonterminal_corrected, default=0.0
        ),
        "other_v62_gate_reproduction_rate": sum(other_v62_checks.values())
        / len(other_v62_checks),
        "maximum_exact_record_delta": float(
            copied_exact_records != source_result["exact_records"]
        ),
        "maximum_official_rollout_record_delta": float(
            copied_rollout_records != source_result["official_rollout_records"]
        ),
        "repair_fixture_pass_rate": implementation_lock["audit_controls"][
            "analytic_fixture_pass_rate"
        ],
        "repair_mutant_kill_rate": implementation_lock["audit_controls"][
            "mutant_kill_rate"
        ],
        "repair_attempt_and_access_violation_count": access_violations,
    }
    qualification = qualify(metrics, config["gates"])
    result = {
        "schema_version": "62r1",
        "experiment": "v62r1_terminal_residual_repair_rescore",
        "evaluation_run": 1,
        "evaluation_implementation_lock": str(
            evaluation_lock_path.relative_to(PROJECT_ROOT)
        ),
        "evaluation_implementation_lock_sha256": file_sha256(evaluation_lock_path),
        "implementation_lock": str(implementation_lock_path.relative_to(PROJECT_ROOT)),
        "implementation_lock_sha256": file_sha256(implementation_lock_path),
        "source_v62_outcome_lock": str(outcome_path.relative_to(PROJECT_ROOT)),
        "source_v62_outcome_lock_sha256": file_sha256(outcome_path),
        "source_v62_result": str(source_result_path.relative_to(PROJECT_ROOT)),
        "source_v62_result_sha256": file_sha256(source_result_path),
        "source_v62_qualification_passed": outcome["qualification_passed"],
        "source_v62_failed_checks": outcome["failed_checks"],
        "external_bundle_seal": str(seal_path.relative_to(PROJECT_ROOT)),
        "external_bundle_seal_sha256": file_sha256(seal_path),
        "source_exact_records_sha256": canonical_sha256(source_result["exact_records"]),
        "source_official_rollout_records_sha256": canonical_sha256(
            source_result["official_rollout_records"]
        ),
        "source_other_v62_gate_checks": other_v62_checks,
        "source_binding_checks": source_binding_checks,
        "node_census": {
            "reachable": reachable_nodes,
            "terminal_support": len(terminal_corrected),
            "nonterminal": len(nonterminal_corrected),
            "old_residual_failures": old_failures,
            "old_terminal_residual_failures": old_terminal_failures,
        },
        "records": records,
        "metrics": metrics,
        "qualification": qualification,
        "data_access": {
            "repair_rescores": 1,
            "new_candidate_evaluations": 0,
            "new_external_rollouts": 0,
            "human_records": 0,
            "model_forward_passes": 0,
        },
        "claim_boundary": config["claimBoundary"],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
