#!/usr/bin/env python3
"""Independently audit and freeze the pre-population V63 design."""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path

import numpy as np

from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT


LEFT, RIGHT, TERMINAL = 2, 3, 4
LISTEN, OPEN_LEFT, OPEN_RIGHT = 0, 1, 2
PERSISTENT, ALTERNATING = 0, 1


def scaled_beta_quadrature(nodes: int, low: float, high: float) -> tuple[np.ndarray, np.ndarray]:
    raw_x, raw_w = np.polynomial.legendre.leggauss(nodes)
    theta = low + (raw_x + 1.0) * (high - low) / 2.0
    unit = (theta - low) / (high - low)
    density = 6.0 * unit * (1.0 - unit) / (high - low)
    weights = raw_w * (high - low) / 2.0 * density
    weights /= weights.sum()
    return theta, weights


def transition_for(base: dict, identity: int, theta: float) -> np.ndarray:
    transition = np.asarray(base["transition"], dtype=np.float64).copy()
    characteristic_same = identity == PERSISTENT
    for source, side in ((0, LEFT), (1, RIGHT), (LEFT, LEFT), (RIGHT, RIGHT)):
        other = RIGHT if side == LEFT else LEFT
        same_probability = theta if characteristic_same else 1.0 - theta
        transition[LISTEN, source] = 0.0
        transition[LISTEN, source, side] = same_probability
        transition[LISTEN, source, other] = 1.0 - same_probability
    return transition


def initial_joint(base: dict, theta_weights: np.ndarray) -> np.ndarray:
    state = np.asarray(base["initial"], dtype=np.float64)
    belief = np.zeros((2, len(theta_weights), len(state)), dtype=np.float64)
    belief[:] = 0.5 * theta_weights[None, :, None] * state[None, None, :]
    return belief


def joint_update(
    base: dict,
    transitions: np.ndarray,
    belief: np.ndarray,
    action: int,
    observation: int,
) -> tuple[np.ndarray, float]:
    obs = np.asarray(base["observation"], dtype=np.float64)[action, :, observation]
    predicted = np.einsum("zks,zkst->zkt", belief, transitions[:, :, action])
    weighted = predicted * obs[None, None, :]
    mass = float(weighted.sum())
    if mass <= 0.0:
        raise ValueError("zero-probability observation")
    return weighted / mass, mass


def expected_rewards(base: dict, transitions: np.ndarray, belief: np.ndarray, action: int) -> float:
    reward = np.asarray(base["reward"], dtype=np.float64)[action]
    return float(np.einsum("zks,zkst,st->", belief, transitions[:, :, action], reward))


def exact_decision(
    base: dict,
    transitions: np.ndarray,
    belief: np.ndarray,
    horizon: int,
) -> tuple[int, float, tuple[float, ...]]:
    if horizon <= 0 or float(belief[..., TERMINAL].sum()) >= 1.0 - 1e-12:
        return LISTEN, 0.0, (0.0, 0.0, 0.0)
    discount = float(base["discount"])
    obs_kernel = np.asarray(base["observation"], dtype=np.float64)
    q_values: list[float] = []
    for action in (LISTEN, OPEN_LEFT, OPEN_RIGHT):
        value = expected_rewards(base, transitions, belief, action)
        if horizon > 1:
            for observation in range(obs_kernel.shape[2]):
                try:
                    posterior, probability = joint_update(
                        base, transitions, belief, action, observation
                    )
                except ValueError:
                    continue
                value += discount * probability * exact_decision(
                    base, transitions, posterior, horizon - 1
                )[1]
        q_values.append(float(value))
    maximum = max(q_values)
    action = next(i for i, value in enumerate(q_values) if maximum - value <= 1e-12)
    return action, float(maximum), tuple(q_values)


def point_model_filter_and_decision(
    base: dict,
    identity: int,
    theta: float,
    observations: tuple[int, ...],
    horizon: int,
) -> int:
    transition = transition_for(base, identity, theta)
    observation = np.asarray(base["observation"], dtype=np.float64)
    reward = np.asarray(base["reward"], dtype=np.float64)
    state_belief = np.asarray(base["initial"], dtype=np.float64).copy()
    for report in observations:
        weighted = (state_belief @ transition[LISTEN]) * observation[LISTEN, :, report]
        state_belief = weighted / weighted.sum()

    def decide(belief: np.ndarray, remaining: int) -> tuple[int, float]:
        if remaining <= 0 or belief[TERMINAL] >= 1.0 - 1e-12:
            return LISTEN, 0.0
        values = []
        for action in (LISTEN, OPEN_LEFT, OPEN_RIGHT):
            value = float(np.einsum("s,st,st->", belief, transition[action], reward[action]))
            if remaining > 1:
                predicted = belief @ transition[action]
                for obs_index in range(observation.shape[2]):
                    weighted = predicted * observation[action, :, obs_index]
                    probability = float(weighted.sum())
                    if probability > 0.0:
                        value += float(base["discount"]) * probability * decide(
                            weighted / probability, remaining - 1
                        )[1]
            values.append(value)
        maximum = max(values)
        action = next(i for i, value in enumerate(values) if maximum - value <= 1e-12)
        return action, float(maximum)

    return decide(state_belief, horizon)[0]


def feasibility(base: dict, config: dict) -> dict:
    family = config["unknownDynamicsFamily"]
    audit_config = config["familyFeasibilityAudit"]
    low, high = family["continuousParameter"]["support"]
    sensor_accuracy = float(family["sensorAccuracy"])
    sensor_factor = (2.0 * sensor_accuracy - 1.0) ** 2
    persistent_range = [sensor_factor * (2.0 * low - 1.0), sensor_factor * (2.0 * high - 1.0)]
    alternating_range = [-persistent_range[1], -persistent_range[0]]
    identity_gap = persistent_range[0] - alternating_range[1]
    theta_slope = 2.0 * sensor_factor

    theta, weights = scaled_beta_quadrature(
        int(audit_config["quadratureNodes"]), float(low), float(high)
    )
    transitions = np.stack(
        [np.stack([transition_for(base, identity, value) for value in theta]) for identity in range(2)]
    )
    transition_normalized = bool(
        np.all(transitions >= 0.0)
        and np.allclose(transitions.sum(axis=4), 1.0, atol=1e-12, rtol=0.0)
    )
    unchanged_actions = bool(
        np.allclose(
            transitions[:, :, OPEN_LEFT],
            np.asarray(base["transition"])[OPEN_LEFT][None, None, :, :],
            atol=0.0,
            rtol=0.0,
        )
        and np.allclose(
            transitions[:, :, OPEN_RIGHT],
            np.asarray(base["transition"])[OPEN_RIGHT][None, None, :, :],
            atol=0.0,
            rtol=0.0,
        )
    )

    report_map = {"tiger-left": 1, "tiger-right": 2}
    history_rows = []
    disagreement_count = 0
    maximum_regret = 0.0
    exact_actions: set[int] = set()
    max_identity_shift = 0.0
    for depth in audit_config["historyDepths"]:
        for symbols in itertools.product(audit_config["historyAlphabet"], repeat=int(depth)):
            reports = tuple(report_map[symbol] for symbol in symbols)
            belief = initial_joint(base, weights)
            history_probability = 1.0
            for report in reports:
                belief, probability = joint_update(base, transitions, belief, LISTEN, report)
                history_probability *= probability
            identity_mass = belief.sum(axis=(1, 2))
            max_identity_shift = max(max_identity_shift, float(np.max(np.abs(identity_mass - 0.5))))
            map_identity = int(np.argmax(identity_mass))
            conditional_theta = belief[map_identity].sum(axis=1)
            conditional_theta /= conditional_theta.sum()
            theta_mean = float(conditional_theta @ theta)
            for horizon in audit_config["remainingHorizons"]:
                exact_action, exact_value, exact_q = exact_decision(
                    base, transitions, belief, int(horizon)
                )
                control_action = point_model_filter_and_decision(
                    base, map_identity, theta_mean, reports, int(horizon)
                )
                regret = exact_value - exact_q[control_action]
                exact_actions.add(exact_action)
                if control_action != exact_action:
                    disagreement_count += 1
                    maximum_regret = max(maximum_regret, float(regret))
                    if len(history_rows) < 12:
                        history_rows.append(
                            {
                                "history": list(symbols),
                                "remaining_horizon": int(horizon),
                                "history_probability": history_probability,
                                "identity_posterior": identity_mass.tolist(),
                                "map_identity": ["persistent", "alternating"][map_identity],
                                "conditional_theta_mean": theta_mean,
                                "exact_action": base["actions"][exact_action],
                                "collapse_action": base["actions"][control_action],
                                "exact_q_values": list(exact_q),
                                "collapse_regret_under_exact_joint_posterior": float(regret),
                            }
                        )

    checks = {
        "transition_normalized": transition_normalized,
        "nonlisten_dynamics_unchanged": unchanged_actions,
        "identity_correlation_gap": identity_gap + 1e-12 >= audit_config["minimumIdentityCorrelationGap"],
        "theta_correlation_slope": theta_slope + 1e-12 >= audit_config["minimumAbsoluteThetaCorrelationSlope"],
        "posterior_identity_shift": max_identity_shift >= audit_config["minimumPosteriorIdentityShift"],
        "collapse_action_disagreements": disagreement_count >= audit_config["minimumExactVersusCollapseActionDisagreements"],
        "collapse_regret": maximum_regret >= audit_config["minimumMaximumCollapseRegret"],
        "exact_uses_open": bool({OPEN_LEFT, OPEN_RIGHT} & exact_actions),
        "exact_uses_listen": LISTEN in exact_actions,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "sensor_factor": sensor_factor,
        "persistent_report_correlation_range": persistent_range,
        "alternating_report_correlation_range": alternating_range,
        "identity_correlation_gap": identity_gap,
        "absolute_theta_correlation_slope": theta_slope,
        "maximum_posterior_identity_shift": max_identity_shift,
        "exact_action_indices_seen": sorted(exact_actions),
        "exact_action_names_seen": [base["actions"][index] for index in sorted(exact_actions)],
        "collapse_action_disagreement_count": disagreement_count,
        "maximum_collapse_regret_under_exact_joint_posterior": maximum_regret,
        "first_disagreements": history_rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/v63-external-unknown-dynamics.json")
    parser.add_argument("--plan", default="docs/v63-external-unknown-dynamics-plan.md")
    parser.add_argument("--audit", default="outputs/v63-external-unknown-dynamics/design-audit.json")
    parser.add_argument("--output", default="configs/v63-design-lock.json")
    args = parser.parse_args()
    config_path = (PROJECT_ROOT / args.config).resolve()
    plan_path = (PROJECT_ROOT / args.plan).resolve()
    audit_path = (PROJECT_ROOT / args.audit).resolve()
    output_path = (PROJECT_ROOT / args.output).resolve()
    if output_path.exists():
        raise RuntimeError("V63 design already frozen")
    config = json.loads(config_path.read_text())
    source_outcome_path = (PROJECT_ROOT / config["sourceV62r1OutcomeLock"]).resolve()
    source_bundle_path = (PROJECT_ROOT / config["sourceV62ExternalBundleSeal"]).resolve()
    source_outcome = json.loads(source_outcome_path.read_text())
    source_bundle = json.loads(source_bundle_path.read_text())
    model_path = (PROJECT_ROOT / config["externalSource"]["sealedModel"]).resolve()
    model = json.loads(model_path.read_text())

    errors: list[str] = []
    source_ok = bool(
        source_outcome["repair_qualification_passed"]
        and not source_outcome["original_v62_qualification_passed"]
        and source_outcome["authorization"]["continue_to_next_preregistered_stage"]
        and not source_outcome["authorization"]["access_human_v58_records"]
        and not source_outcome["authorization"]["model_access"]
    )
    if not source_ok:
        errors.append("V62r1 source authorization or failure boundary is not intact")
    bundle_ok = bool(
        file_sha256(model_path) == config["externalSource"]["sealedModelSha256"]
        and source_bundle["source_files_sha256"][config["externalSource"]["model"]]
        == config["externalSource"]["sourceSha256"]
        and not source_bundle["authorization"]["modify_external_bundle"]
    )
    if not bundle_ok:
        errors.append("pinned external Tiger source or sealed model hash is not intact")
    family = feasibility(model, config)
    if not family["passed"]:
        errors.append("unknown-dynamics family failed preregistered feasibility gates")
    inherited = config["smcSquared"]
    inherited_ok = bool(
        inherited["outerThetaParticleBudgets"] == [31, 127, 509]
        and inherited["independentRepeatsOnExactBenchmark"] == 3
        and inherited["innerStateParticleBudget"] == 127
        and inherited["outerEssThresholdFraction"] == 0.5
        and inherited["innerEssThresholdFraction"] == 0.5
        and inherited["rejuvenationStepsPerOuterResampling"] == 2
        and inherited["proposalStandardDeviation"] == 0.4
    )
    if not inherited_ok:
        errors.append("V63 does not preserve the declared frozen V53r2 SMC2 settings")
    downstream = (
        "configs/v63-design-lock.json",
        "configs/v63-implementation-lock.json",
        "configs/v63-population-seal.json",
        "configs/v63-evaluation-implementation-lock.json",
        "configs/v63-outcome-lock.json",
        "data/v63-external-unknown-dynamics/sealed-populations",
        "data/v63-external-unknown-dynamics/manifest.json",
    )
    downstream_absent = not any((PROJECT_ROOT / value).exists() for value in downstream)
    if not downstream_absent:
        errors.append("V63 downstream implementation, population, or outcome already exists")
    boundary_ok = bool(
        not config["claimBoundary"]["upstreamSuppliesUnknownDynamicsFamily"]
        and config["claimBoundary"]["projectAddsPreregisteredUncertaintyLayer"]
        and not config["claimBoundary"]["activeInterventionSelection"]
        and not config["claimBoundary"]["multipleInformativeActions"]
        and config["firewall"]["activeSelectionOnTigerAsSubstantiveEIGTest"] == "forbidden"
    )
    if not boundary_ok:
        errors.append("external-source or Tiger active-design claim boundary is missing")

    audit = {
        "schema_version": 63,
        "experiment": "v63_prepopulation_design_audit",
        "passed": not errors,
        "decision": "authorize_v63_design_lock" if not errors else "reject_or_repair_v63_design",
        "errors": errors,
        "checks": {
            "source_v62r1_boundary_and_authorization": source_ok,
            "external_bundle_and_model_hashes": bundle_ok,
            "family_feasibility": family["passed"],
            "frozen_v53r2_smc2_settings_inherited": inherited_ok,
            "downstream_absent": downstream_absent,
            "external_and_active_design_claim_boundaries": boundary_ok,
        },
        "family_feasibility": family,
        "bindings": {
            "config": str(config_path.relative_to(PROJECT_ROOT)),
            "config_sha256": file_sha256(config_path),
            "preregistration": str(plan_path.relative_to(PROJECT_ROOT)),
            "preregistration_sha256": file_sha256(plan_path),
            "source_v62r1_outcome_lock": str(source_outcome_path.relative_to(PROJECT_ROOT)),
            "source_v62r1_outcome_lock_sha256": file_sha256(source_outcome_path),
            "source_v62_bundle_seal": str(source_bundle_path.relative_to(PROJECT_ROOT)),
            "source_v62_bundle_seal_sha256": file_sha256(source_bundle_path),
            "sealed_tiger_model": str(model_path.relative_to(PROJECT_ROOT)),
            "sealed_tiger_model_sha256": file_sha256(model_path),
        },
        "data_access": {
            "candidate_population_records_accessed": 0,
            "candidate_evaluation_runs": 0,
            "human_record_access_count": 0,
            "simulated_human_record_count": 0,
            "model_forward_pass_count": 0,
            "adapter_training_run_count": 0,
        },
    }
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    if errors:
        print(json.dumps(audit, indent=2, sort_keys=True))
        raise SystemExit(1)

    locked_config = json.loads(json.dumps(config))
    locked_config["stageAuthorization"] = {
        "auditAndFreezeDesign": False,
        "writeAndAuditImplementation": True,
        "constructSealedPopulations": False,
        "runOneCandidateEvaluation": False,
        "activeInterventionSelection": False,
        "rewardOrPlanningEvaluation": False,
        "modelAccess": False,
    }
    lock = {
        "schema_version": 63,
        "experiment": "v63_design_lock",
        "config": str(config_path.relative_to(PROJECT_ROOT)),
        "config_sha256": file_sha256(config_path),
        "preregistration": str(plan_path.relative_to(PROJECT_ROOT)),
        "preregistration_sha256": file_sha256(plan_path),
        "design_audit": str(audit_path.relative_to(PROJECT_ROOT)),
        "design_audit_sha256": file_sha256(audit_path),
        "source_v62r1_outcome_lock": str(source_outcome_path.relative_to(PROJECT_ROOT)),
        "source_v62r1_outcome_lock_sha256": file_sha256(source_outcome_path),
        "source_v62_bundle_seal": str(source_bundle_path.relative_to(PROJECT_ROOT)),
        "source_v62_bundle_seal_sha256": file_sha256(source_bundle_path),
        "config_payload": locked_config,
        "family_feasibility_summary": {
            "identity_correlation_gap": family["identity_correlation_gap"],
            "absolute_theta_correlation_slope": family["absolute_theta_correlation_slope"],
            "maximum_posterior_identity_shift": family["maximum_posterior_identity_shift"],
            "collapse_action_disagreement_count": family["collapse_action_disagreement_count"],
            "maximum_collapse_regret_under_exact_joint_posterior": family["maximum_collapse_regret_under_exact_joint_posterior"],
            "exact_action_names_seen": family["exact_action_names_seen"],
        },
        "authorization": locked_config["stageAuthorization"],
    }
    lock["lock_payload_sha256"] = hashlib.sha256(
        json.dumps(lock, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    output_path.write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"audit": audit, "lock": lock}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
