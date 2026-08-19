from __future__ import annotations

from fractions import Fraction
from typing import Any

from v165_factored_ontology_identifiability_population import candidate_universe
from v174_certificate_depth_feasibility_census import minimal_target_certificate
from v175_certification_aware_planner_development import _from_payload
from v180_triple_repetition_robust_planner_development import (
    evaluate_development,
    evaluate_safety_gates as evaluate_V180_safety_gates,
)


def build_certificate_artifact(
    states_artifact: dict[str, Any], targets_artifact: dict[str, Any]
) -> dict[str, Any]:
    universe = {row["candidate_id"]: row for row in candidate_universe()}
    states = {row["state_id"]: row for row in states_artifact["states"]}
    rows = []
    for target in targets_artifact["target_cases"]:
        state = states[target["state_id"]]
        certificate = minimal_target_certificate(
            state["candidate_ids"], target["target_candidate_id"], universe
        )
        rows.append(
            {
                **target,
                **certificate,
                "raw_inspection_count": 3 * certificate["minimal_depth"],
                "certificate_valid": certificate["certified_class"]
                == target["target_class"],
            }
        )
    rows.sort(key=lambda row: row["target_case_id"])
    return {"target_results": rows}


def evaluate_confirmation(
    states_artifact: dict[str, Any],
    eligible_artifact: dict[str, Any],
    targets_artifact: dict[str, Any],
    planner_config: dict[str, Any],
    sandbox_config: dict[str, Any],
    horizon: int,
    block_cost: Fraction,
    clean_cost: Fraction,
) -> dict[str, Any]:
    certificates = build_certificate_artifact(states_artifact, targets_artifact)
    evaluation = evaluate_development(
        states_artifact,
        eligible_artifact,
        targets_artifact,
        certificates,
        planner_config,
        sandbox_config,
        horizon,
        block_cost,
        clean_cost,
    )
    validity = sum(
        row["certificate_valid"] for row in certificates["target_results"]
    ) / len(certificates["target_results"])
    evaluation["summary"]["oracle_certificate_validity_rate"] = validity
    depths = sorted(
        {row["minimal_depth"] for row in certificates["target_results"]}
    )
    evaluation["certificate_digest"] = {
        "target_count": len(certificates["target_results"]),
        "validity_rate": validity,
        "minimal_block_depth_counts": {
            str(depth): sum(
                row["minimal_depth"] == depth
                for row in certificates["target_results"]
            )
            for depth in depths
        },
        "minimal_raw_inspection_counts": {
            str(3 * depth): sum(
                row["minimal_depth"] == depth
                for row in certificates["target_results"]
            )
            for depth in depths
        },
    }
    return evaluation


def evaluate_safety_gates(
    evaluation: dict[str, Any], access: dict[str, int], config: dict[str, Any]
) -> dict[str, bool]:
    checks = evaluate_V180_safety_gates(evaluation, access, config)
    checks["oracle_certificate_validity"] = (
        evaluation["summary"]["oracle_certificate_validity_rate"]
        == config["integrityAndSafetyGates"]["requiredOracleCertificateValidity"]
    )
    return checks


def evaluate_primary_confirmation(
    evaluation: dict[str, Any], config: dict[str, Any]
) -> dict[str, bool]:
    summary = evaluation["summary"]
    metrics = summary["policy_metrics"]
    exact = _from_payload(
        metrics["exact_robust_certification_adaptive"]["routed_total_risk"]
    )
    return {
        "below_immediate_defer": exact
        < _from_payload(metrics["immediate_defer"]["routed_total_risk"]),
        "positive_trusted_completion": _from_payload(
            metrics["exact_robust_certification_adaptive"]["trusted_completion"]
        )
        > 0,
        "statewise_improvement_exists": summary[
            "state_count_strictly_improved_over_immediate_defer"
        ]
        >= config["primaryConfirmationThresholds"][
            "minimumStateCountStrictlyImprovedOverImmediateDefer"
        ],
    }


def evaluate_strong_confirmation(
    evaluation: dict[str, Any], config: dict[str, Any]
) -> dict[str, bool]:
    summary = evaluation["summary"]
    metrics = summary["policy_metrics"]
    exact = _from_payload(
        metrics["exact_robust_certification_adaptive"]["routed_total_risk"]
    )
    return {
        "no_worse_than_repriced_clean": exact
        <= _from_payload(
            metrics["V175_clean_policy_repriced_at_triple_cost"][
                "routed_total_risk"
            ]
        ),
        "no_worse_than_greedy": exact
        <= _from_payload(
            metrics["greedy_information_gain_triple_blocks"][
                "routed_total_risk"
            ]
        ),
        "no_worse_than_open_loop": exact
        <= _from_payload(
            metrics["optimal_open_loop_triple_blocks"]["routed_total_risk"]
        ),
        "no_worse_than_random": exact
        <= _from_payload(
            metrics["random_block_order_consensus_stop"]["routed_total_risk"]
        ),
        "pointwise_no_worse_every_control": summary[
            "statewise_no_worse_than_every_operational_control_rate"
        ]
        >= config["strongConfirmationThresholds"][
            "requiredStatewiseNoWorseThanEveryOperationalControlRate"
        ],
    }


__all__ = [
    "build_certificate_artifact",
    "evaluate_confirmation",
    "evaluate_primary_confirmation",
    "evaluate_safety_gates",
    "evaluate_strong_confirmation",
]
