#!/usr/bin/env python3
"""Independently audit and freeze the pre-population V64 design."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT
from v62_external_pomdp import POMDPModel, parse_pomdp_file


def quadrature(nodes: int, low: float, high: float) -> tuple[np.ndarray, np.ndarray]:
    raw_x, raw_w = np.polynomial.legendre.leggauss(nodes)
    theta = low + (raw_x + 1.0) * (high - low) / 2.0
    unit = (theta - low) / (high - low)
    density = 6.0 * unit * (1.0 - unit) / (high - low)
    weights = raw_w * (high - low) / 2.0 * density
    weights /= weights.sum()
    return theta, weights


def action_structure(model: POMDPModel) -> tuple[list[int], list[np.ndarray]]:
    index = {name: i for i, name in enumerate(model.actions)}
    canonical = [index[name] for name in ("n", "e", "s", "w")]
    clockwise = np.asarray(
        [index["e"], index["w"], index["s"], index["n"]], dtype=np.int64
    )
    counterclockwise = np.asarray(
        [index["w"], index["e"], index["n"], index["s"]], dtype=np.int64
    )
    return canonical, [clockwise, counterclockwise]


def family_transitions(
    model: POMDPModel, theta: np.ndarray, permutations: list[np.ndarray]
) -> np.ndarray:
    return np.asarray(
        [
            [
                [
                    value * model.transition[action]
                    + (1.0 - value) * model.transition[permutation[action]]
                    for action in range(len(model.actions))
                ]
                for value in theta
            ]
            for permutation in permutations
        ],
        dtype=np.float64,
    )


def initial_belief(
    model: POMDPModel, prior: np.ndarray, observation: int
) -> tuple[np.ndarray | None, float]:
    joint = (
        prior[:, :, None]
        * model.initial[None, None, :]
        * model.observation[0, :, observation][None, None, :]
    )
    probability = float(joint.sum())
    if probability <= 1e-15:
        return None, probability
    return joint / probability, probability


def predict(
    model: POMDPModel, transitions: np.ndarray, belief: np.ndarray, action: int
) -> tuple[np.ndarray, np.ndarray]:
    state_prediction = np.einsum(
        "zqs,zqst->zqt", belief, transitions[:, :, action]
    )
    joint_parameter_observation = np.einsum(
        "zqs,so->zqo", state_prediction, model.observation[action]
    )
    return state_prediction, joint_parameter_observation


def update(
    model: POMDPModel,
    transitions: np.ndarray,
    belief: np.ndarray,
    action: int,
    observation: int,
) -> tuple[np.ndarray, float]:
    state_prediction, joint = predict(model, transitions, belief, action)
    probability = float(joint[:, :, observation].sum())
    weighted = (
        state_prediction
        * model.observation[action, :, observation][None, None, :]
    )
    return weighted / probability, probability


def expected_information_gain(
    model: POMDPModel, transitions: np.ndarray, belief: np.ndarray, action: int
) -> float:
    _, joint = predict(model, transitions, belief, action)
    parameter = belief.sum(axis=2)
    conditional = np.divide(
        joint,
        parameter[:, :, None],
        out=np.zeros_like(joint),
        where=parameter[:, :, None] > 0.0,
    )
    marginal = joint.sum(axis=(0, 1))
    ratio = np.divide(
        conditional,
        marginal[None, None, :],
        out=np.ones_like(conditional),
        where=marginal[None, None, :] > 0.0,
    )
    mask = (joint > 0.0) & (ratio > 0.0)
    return float(np.sum(joint[mask] * np.log(ratio[mask])))


def identity_information(
    model: POMDPModel, transitions: np.ndarray, belief: np.ndarray, action: int
) -> float:
    _, joint_parameter_observation = predict(model, transitions, belief, action)
    joint = joint_parameter_observation.sum(axis=1)
    identity = belief.sum(axis=(1, 2))
    conditional = np.divide(
        joint,
        identity[:, None],
        out=np.zeros_like(joint),
        where=identity[:, None] > 0.0,
    )
    marginal = joint.sum(axis=0)
    ratio = np.divide(
        conditional,
        marginal[None, :],
        out=np.ones_like(conditional),
        where=marginal[None, :] > 0.0,
    )
    mask = (joint > 0.0) & (ratio > 0.0)
    return float(np.sum(joint[mask] * np.log(ratio[mask])))


def feasibility_census(
    model: POMDPModel,
    transitions: np.ndarray,
    prior: np.ndarray,
    canonical: list[int],
    depths: list[int],
    strict_margin: float,
) -> dict:
    front: list[tuple[np.ndarray, float]] = []
    for observation in range(len(model.observations)):
        belief, probability = initial_belief(model, prior, observation)
        if belief is not None:
            front.append((belief, probability))
    positive_actions: set[int] = set()
    strict_actions: set[int] = set()
    depth_rows: list[dict] = []
    identity_fraction_by_action: dict[int, list[float]] = {
        action: [] for action in range(len(model.actions))
    }
    for depth in range(max(depths) + 1):
        weighted_oracle = weighted_random = weighted_fixed = total = 0.0
        next_front: list[tuple[np.ndarray, float]] = []
        for belief, reach in front:
            eig = np.asarray(
                [
                    expected_information_gain(model, transitions, belief, action)
                    for action in range(len(model.actions))
                ]
            )
            for action, value in enumerate(eig):
                if value > 1e-8:
                    positive_actions.add(action)
                    identity_fraction_by_action[action].append(
                        identity_information(model, transitions, belief, action) / value
                    )
            order = np.argsort(eig)
            if eig[order[-1]] - eig[order[-2]] > strict_margin:
                strict_actions.add(int(order[-1]))
            weighted_oracle += reach * float(eig.max())
            weighted_random += reach * float(eig.mean())
            weighted_fixed += reach * float(eig[canonical[depth % 4]])
            total += reach
            if depth < max(depths):
                for action in range(len(model.actions)):
                    _, joint = predict(model, transitions, belief, action)
                    probabilities = joint.sum(axis=(0, 1))
                    for observation, probability in enumerate(probabilities):
                        child_reach = reach * float(probability) / len(model.actions)
                        if child_reach <= 1e-12:
                            continue
                        child, _ = update(
                            model, transitions, belief, action, observation
                        )
                        next_front.append((child, child_reach))
        if depth in depths:
            depth_rows.append(
                {
                    "depth": depth,
                    "reachable_history_nodes": len(front),
                    "probability_mass": total,
                    "mean_oracle_eig": weighted_oracle / total,
                    "mean_uniform_random_eig": weighted_random / total,
                    "mean_fixed_cycle_eig": weighted_fixed / total,
                }
            )
        front = next_front
    return {
        "depth_rows": depth_rows,
        "positive_information_actions": [model.actions[a] for a in sorted(positive_actions)],
        "strict_eig_maximizers": [model.actions[a] for a in sorted(strict_actions)],
        "equal_depth_mean_oracle_minus_random": float(
            np.mean(
                [
                    row["mean_oracle_eig"] - row["mean_uniform_random_eig"]
                    for row in depth_rows
                ]
            )
        ),
        "equal_depth_mean_oracle_minus_fixed": float(
            np.mean(
                [
                    row["mean_oracle_eig"] - row["mean_fixed_cycle_eig"]
                    for row in depth_rows
                ]
            )
        ),
        "identity_information_fraction_ranges": {
            model.actions[action]: [float(min(values)), float(max(values))]
            for action, values in identity_fraction_by_action.items()
            if values
        },
    }


def known_model_q(
    model: POMDPModel,
    permutation: np.ndarray,
    theta: float,
    horizon: int,
) -> np.ndarray:
    transition = np.asarray(
        [
            theta * model.transition[action]
            + (1.0 - theta) * model.transition[permutation[action]]
            for action in range(len(model.actions))
        ]
    )
    value = np.zeros(len(model.states), dtype=np.float64)
    q_values = np.zeros((len(model.actions), len(model.states)), dtype=np.float64)
    for _ in range(horizon):
        q_values = np.asarray(
            [
                np.sum(
                    transition[action]
                    * (
                        model.reward[action]
                        + model.discount * value[None, :]
                    ),
                    axis=1,
                )
                for action in range(len(model.actions))
            ]
        )
        value = q_values.max(axis=0)
    return q_values


def decision_disagreement_states(
    model: POMDPModel,
    permutations: list[np.ndarray],
    low: float,
    high: float,
    horizon: int,
) -> list[int]:
    actions = []
    for permutation in permutations:
        for theta in (low, (low + high) / 2.0, high):
            actions.append(np.argmax(known_model_q(model, permutation, theta, horizon), axis=0))
    return [
        state
        for state in range(len(model.states))
        if len({int(action[state]) for action in actions}) > 1
    ]


def posterior_kl(belief: np.ndarray, prior: np.ndarray) -> float:
    parameter = belief.sum(axis=2)
    mask = parameter > 0.0
    return float(np.sum(parameter[mask] * np.log(parameter[mask] / prior[mask])))


def sample_categorical(probability: np.ndarray, uniform: float) -> int:
    return min(
        len(probability) - 1,
        int(np.searchsorted(np.cumsum(probability), uniform, side="right")),
    )


def adaptive_feasibility_pilot(
    model: POMDPModel,
    transitions: np.ndarray,
    prior: np.ndarray,
    permutations: list[np.ndarray],
    canonical: list[int],
    low: float,
    high: float,
    replications: int,
    seed: int,
) -> dict:
    root = np.random.SeedSequence(seed)
    latent_rng = np.random.default_rng(root.spawn(1)[0])
    policies = ("adaptive_eig", "fixed", "random")
    transition_rng = {
        policy: np.random.default_rng(root.spawn(1)[0]) for policy in policies
    }
    random_action_rng = np.random.default_rng(root.spawn(1)[0])
    information = {policy: [] for policy in policies}
    action_counts = {policy: np.zeros(len(model.actions), dtype=np.int64) for policy in policies}
    for replication in range(replications):
        identity = replication % 2
        theta = low + (high - low) * latent_rng.beta(2.0, 2.0)
        initial_state = sample_categorical(model.initial, latent_rng.random())
        initial_observation = int(np.argmax(model.observation[0, initial_state]))
        for policy in policies:
            state = initial_state
            belief, _ = initial_belief(model, prior, initial_observation)
            if belief is None:
                raise RuntimeError("pilot sampled impossible initial observation")
            for step in range(8):
                if policy == "adaptive_eig":
                    values = np.asarray(
                        [
                            expected_information_gain(
                                model, transitions, belief, action
                            )
                            for action in range(len(model.actions))
                        ]
                    )
                    maximum = float(values.max())
                    action = next(
                        candidate
                        for candidate in canonical
                        if values[candidate] >= maximum - 1e-12
                    )
                elif policy == "fixed":
                    action = canonical[step % 4]
                else:
                    action = canonical[
                        min(3, int(random_action_rng.random() * len(canonical)))
                    ]
                action_counts[policy][action] += 1
                true_transition = (
                    theta * model.transition[action]
                    + (1.0 - theta)
                    * model.transition[permutations[identity][action]]
                )
                state = sample_categorical(
                    true_transition[state], transition_rng[policy].random()
                )
                observation = int(np.argmax(model.observation[action, state]))
                belief, _ = update(model, transitions, belief, action, observation)
            information[policy].append(posterior_kl(belief, prior))
    summary = {
        "replications": replications,
        "seed": seed,
        "budget": 8,
        "mean_information_nats": {
            policy: float(np.mean(values)) for policy, values in information.items()
        },
        "action_counts": {
            policy: {
                model.actions[action]: int(count)
                for action, count in enumerate(counts)
            }
            for policy, counts in action_counts.items()
        },
        "paired_differences": {},
    }
    adaptive = np.asarray(information["adaptive_eig"])
    for baseline in ("fixed", "random"):
        difference = adaptive - np.asarray(information[baseline])
        standard_error = float(difference.std(ddof=1) / np.sqrt(replications))
        summary["paired_differences"][f"adaptive_minus_{baseline}"] = {
            "mean": float(difference.mean()),
            "standard_error": standard_error,
            "normal_lower_95": float(difference.mean() - 1.96 * standard_error),
        }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/v64-external-multi-action-eig.json")
    parser.add_argument("--plan", default="docs/v64-external-multi-action-eig-plan.md")
    parser.add_argument(
        "--audit", default="outputs/v64-external-multi-action-eig/design-audit.json"
    )
    parser.add_argument("--output", default="configs/v64-design-lock.json")
    args = parser.parse_args()
    config_path = (PROJECT_ROOT / args.config).resolve()
    plan_path = (PROJECT_ROOT / args.plan).resolve()
    audit_path = (PROJECT_ROOT / args.audit).resolve()
    output_path = (PROJECT_ROOT / args.output).resolve()
    if output_path.exists():
        raise RuntimeError("V64 design already frozen")
    config = json.loads(config_path.read_text())
    errors: list[str] = []

    source_path = (PROJECT_ROOT / config["sourceV63r1OutcomeLock"]).resolve()
    source = json.loads(source_path.read_text())
    source_ok = bool(
        source["qualification_passed"]
        and source["authorization"]["preregister_separate_multi_action_external_EIG_stage"]
        and not source["authorization"]["construct_or_run_EIG_population"]
        and not source["authorization"]["reward_or_planning_evaluation"]
        and file_sha256(PROJECT_ROOT / source["result"]) == source["result_sha256"]
        and file_sha256(PROJECT_ROOT / source["summary"]) == source["summary_sha256"]
        and file_sha256(PROJECT_ROOT / source["post_result_audit"])
        == source["post_result_audit_sha256"]
    )
    if not source_ok:
        errors.append("V63r1 does not bind or authorize V64 preregistration")

    external = config["externalAnchor"]
    model_path = (PROJECT_ROOT / external["workspaceModelPath"]).resolve()
    model = parse_pomdp_file(model_path)
    external_ok = bool(
        file_sha256(model_path) == external["modelSha256"]
        and external["commit"] == "a5e1d62d14e4efe783885b9d4f19cffa2a568eec"
        and len(model.states) == external["expectedStates"] == 11
        and list(model.actions) == external["expectedActions"]
        and list(model.observations) == external["expectedObservations"]
        and model.discount == external["discount"]
        and model_path.name == "4x3_nonterminating.POMDP"
    )
    if not external_ok:
        errors.append("pinned external 4x3 model binding or shape is invalid")

    family = config["unknownDynamicsFamily"]
    family_ok = bool(
        family["identityNames"] == ["clockwise_failure", "counterclockwise_failure"]
        and family["identityPrior"] == [0.5, 0.5]
        and family["thetaSupport"] == [0.6, 0.95]
        and family["thetaPrior"] == "scaled_beta_2_2"
        and family["canonicalActionOrder"] == ["n", "e", "s", "w"]
        and family["projectAuthoredLayer"]
        and len(family["unchangedSourceArrays"]) == 4
    )
    if not family_ok:
        errors.append("unknown actuator family or source/project boundary is invalid")

    boundary = config["claimBoundary"]
    boundary_ok = bool(
        boundary["externalActiveSystemIdentification"]
        and boundary["fourCompleteNonterminalInterventions"]
        and boundary["exactHistoryDependentBelief"]
        and boundary["exactStaticLatentExpectedInformationGain"]
        and boundary["matchedFixedAndRandomDesignComparison"]
        and boundary["targetLatent"]
        == "joint_actuator_identity_and_continuous_theta"
        and boundary["mazeStateIntegratedAsNuisance"]
        and not any(
            boundary[key]
            for key in (
                "rewardPlanning",
                "multiStepBayesOptimalExperimentalDesign",
                "approximateParticleAcquisition",
                "learnedAcquisition",
                "formalVerification",
                "languageGrounding",
                "humanData",
                "modelAccess",
                "adapterTraining",
            )
        )
    )
    if not boundary_ok:
        errors.append("V64 exact acquisition claim boundary is too broad or incomplete")

    policies = config["designPolicies"]
    policies_ok = bool(
        policies["interactionBudgets"] == [1, 2, 4, 6, 8]
        and policies["samePriorInitialDistributionAndBudget"]
        and policies["pairedLatents"]
        and len({policies["adaptiveEIG"], policies["fixed"], policies["random"]}) == 3
    )
    population = config["selectionPopulation"]
    adaptive = config["pairedAdaptiveEvaluation"]
    sbc = config["adaptiveSBC"]
    populations_ok = bool(
        population["records"] == 192
        and population["recordsPerPublicPrefixLength"]
        * len(population["publicPrefixLengths"])
        == population["records"]
        and population["retainEveryGeneratedRecord"]
        and population["noSelectionByEIGSpreadOrOptimalAction"]
        and adaptive["replications"] == 512
        and adaptive["identityBalance"] == "exact_256_per_identity_by_frozen_replication_parity"
        and adaptive["policies"] == ["adaptiveEIG", "fixed", "random"]
        and adaptive["primaryBudget"] == 8
        and sbc["replications"] == 256
        and sbc["rankSupportSize"] == sbc["posteriorDrawsPerReplication"] + 1
        and sbc["replications"] / sbc["rankBins"] == sbc["expectedRanksPerBin"]
        and sbc["selectionUsesOnlyPreOutcomePublicHistory"]
    )
    if not policies_ok or not populations_ok:
        errors.append("policy comparison or future population design is inconsistent")

    controls = config["controls"]
    gates = config["gates"]
    control_count = len([key for key in controls if key.endswith("Control")])
    gates_ok = bool(
        control_count == 6
        and controls["minimumDetectedOrDominated"]
        == gates["minimumControlsDetectedOrDominated"]
        == 5
        and gates["minimumOptimalSetMembershipRate"] == 1.0
        and gates["maximumSelectedEIGRegretNats"] == 1e-10
        and gates["minimumDistinctStrictlyOptimalCommands"] == 2
        and gates["maximumDominantCommandSelectionRate"] == 0.9
        and gates["minimumMeanOracleMinusUniformRandomEIGNats"] == 0.003
        and gates["minimumMeanOracleMinusFixedCycleEIGNats"] == 0.002
        and gates["minimumBudget8AdaptiveMinusFixedInformationLower95Nats"] == 0.02
        and gates["minimumBudget8AdaptiveMinusRandomInformationLower95Nats"] == 0.02
        and all(
            gates[key] == 0
            for key in (
                "maximumTruthFieldAccessCount",
                "maximumRealizedOutcomeAccessBeforeSelectionCount",
                "maximumCandidateOmissionCount",
                "maximumTieBreakViolationCount",
                "maximumRandomStreamCollisionCount",
            )
        )
    )
    if not gates_ok:
        errors.append("V64 controls or noncompensatory gates are inconsistent")

    stage = config["stageAuthorization"]
    firewall_ok = bool(
        set(config["firewall"].values()) == {"forbidden"}
        and stage["writeAndAuditExactEIGImplementation"]
        and not any(value for key, value in stage.items() if key != "writeAndAuditExactEIGImplementation")
    )
    if not firewall_ok:
        errors.append("V64 firewall or design-only stage authorization is invalid")

    seeds: list[int] = []
    for section in (config["selectionPopulation"], adaptive, sbc):
        seeds.extend(
            value
            for key, value in section.items()
            if key.endswith("Seed") and isinstance(value, int)
        )
    seeds.append(config["preRegistrationFeasibilityAudit"]["adaptivePilotSeed"])
    seeds_ok = len(seeds) == len(set(seeds))
    if not seeds_ok:
        errors.append("V64 root seeds are not distinct")

    canonical, permutations = action_structure(model)
    feasibility = config["preRegistrationFeasibilityAudit"]
    low, high = family["thetaSupport"]
    theta, theta_weights = quadrature(feasibility["quadratureNodes"], low, high)
    prior = np.stack([0.5 * theta_weights, 0.5 * theta_weights])
    transitions = family_transitions(model, theta, permutations)
    row_error = float(np.max(np.abs(transitions.sum(axis=-1) - 1.0)))
    minimum_probability = float(transitions.min())
    census = feasibility_census(
        model,
        transitions,
        prior,
        canonical,
        feasibility["reachableHistoryCensusDepths"],
        feasibility["strictMarginNats"],
    )
    disagreement_states = decision_disagreement_states(
        model,
        permutations,
        low,
        high,
        feasibility["decisionRelevanceHorizon"],
    )
    pilot = adaptive_feasibility_pilot(
        model,
        transitions,
        prior,
        permutations,
        canonical,
        low,
        high,
        feasibility["adaptivePilotReplications"],
        feasibility["adaptivePilotSeed"],
    )
    feasibility_ok = bool(
        row_error <= 1e-12
        and minimum_probability >= -1e-15
        and len(census["positive_information_actions"])
        >= feasibility["minimumPositiveInformationCommands"]
        and len(census["strict_eig_maximizers"])
        >= feasibility["minimumDistinctStrictEIGMaximizers"]
        and census["equal_depth_mean_oracle_minus_random"]
        >= feasibility["minimumMeanOracleMinusRandomNats"]
        and census["equal_depth_mean_oracle_minus_fixed"]
        >= feasibility["minimumMeanOracleMinusFixedNats"]
        and len(disagreement_states)
        >= feasibility["minimumKnownStateDecisionDisagreementStates"]
        and pilot["paired_differences"]["adaptive_minus_fixed"]["normal_lower_95"]
        >= feasibility["minimumPilotBudget8AdaptiveMinusFixedLower95Nats"]
        and pilot["paired_differences"]["adaptive_minus_random"]["normal_lower_95"]
        >= feasibility["minimumPilotBudget8AdaptiveMinusRandomLower95Nats"]
    )
    if not feasibility_ok:
        errors.append("pre-registration family feasibility audit failed")

    downstream = (
        "configs/v64-implementation-lock.json",
        "configs/v64-population-seal.json",
        "configs/v64-evaluation-implementation-lock.json",
        "configs/v64-outcome-lock.json",
        "data/v64-external-multi-action-eig",
        "outputs/v64-external-multi-action-eig/implementation-audit.json",
        "outputs/v64-external-multi-action-eig/evaluation",
    )
    downstream_absent = not any((PROJECT_ROOT / path).exists() for path in downstream)
    if not downstream_absent:
        errors.append("V64 downstream artifacts exist before the design lock")

    audit = {
        "schema_version": 64,
        "experiment": "v64_design_audit",
        "passed": not errors,
        "decision": "authorize_v64_design_lock" if not errors else "repair_v64_design",
        "errors": errors,
        "checks": {
            "source_v63r1_authorization_and_binding": source_ok,
            "pinned_external_model": external_ok,
            "project_authored_unknown_dynamics_boundary": family_ok,
            "exact_active_identification_claim_boundary": boundary_ok,
            "matched_policy_and_future_population_design": policies_ok and populations_ok,
            "controls_and_noncompensatory_gates": gates_ok,
            "design_only_firewall": firewall_ok,
            "distinct_root_seeds": seeds_ok,
            "family_feasibility": feasibility_ok,
            "downstream_absent": downstream_absent,
        },
        "family_feasibility": {
            "maximum_transition_row_error": row_error,
            "minimum_transition_probability": minimum_probability,
            "census": census,
            "known_state_horizon_5_decision_disagreement_states": disagreement_states,
            "adaptive_budget_8_pilot": pilot,
            "pilot_is_not_evaluation_population": True,
        },
        "data_access": {
            "v64_evaluation_population_records": 0,
            "v64_evaluation_runs": 0,
            "v64_SBC_runs": 0,
            "human_records": 0,
            "simulated_human_records": 0,
            "model_forward_passes": 0,
            "adapter_training_runs": 0,
        },
    }
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    if errors:
        print(json.dumps(audit, indent=2, sort_keys=True))
        raise SystemExit(1)

    lock = {
        "schema_version": 64,
        "experiment": "v64_design_lock",
        "config": str(config_path.relative_to(PROJECT_ROOT)),
        "config_sha256": file_sha256(config_path),
        "config_payload": config,
        "preregistration": str(plan_path.relative_to(PROJECT_ROOT)),
        "preregistration_sha256": file_sha256(plan_path),
        "source_v63r1_outcome_lock": str(source_path.relative_to(PROJECT_ROOT)),
        "source_v63r1_outcome_lock_sha256": file_sha256(source_path),
        "external_model": str(model_path.relative_to(PROJECT_ROOT)),
        "external_model_sha256": file_sha256(model_path),
        "design_audit": str(audit_path.relative_to(PROJECT_ROOT)),
        "design_audit_sha256": file_sha256(audit_path),
        "family_feasibility_summary": audit["family_feasibility"],
        "authorization": {
            "modify_design": False,
            "write_and_audit_exact_EIG_implementation": True,
            "construct_selection_population": False,
            "construct_adaptive_population": False,
            "run_evaluation": False,
            "approximate_particle_acquisition": False,
            "reward_planning": False,
            "formal_verification": False,
            "access_human_data": False,
            "simulate_human_data": False,
            "model_access": False,
            "adapter_training": False,
        },
    }
    lock["lock_payload_sha256"] = hashlib.sha256(
        json.dumps(lock, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    output_path.write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"audit": audit, "lock": lock}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
