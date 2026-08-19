from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Sequence

import numpy as np


LATENT_NAMES = ("CANONICAL", "REVERSED", "OUTSIDE_UNKNOWN")
CONDITION_NAMES = ("A", "B")
OBSERVATION_NAMES = ("red", "blue", "green")
STAGE_NAMES = ("PRE_CALIBRATION", "POST_CALIBRATION", "POST_INSPECTION")
ACTION_NAMES = ("calibrate", "inspect", "repair_A", "repair_B", "defer")
SENSING_ACTIONS = ("calibrate", "inspect")
REPAIR_ACTIONS = ("repair_A", "repair_B")


@dataclass(frozen=True)
class FixedStageKernel:
    calibration: np.ndarray
    inspection: np.ndarray
    sensing_cost: float
    deferral_reward: float
    repair_immediate_reward: float
    correct_settlement_reward: float
    wrong_settlement_reward: float
    discount: float

    def __post_init__(self) -> None:
        calibration = np.asarray(self.calibration, dtype=np.float64)
        inspection = np.asarray(self.inspection, dtype=np.float64)
        if calibration.ndim != 2 or calibration.shape[0] < 1 or calibration.shape[1] != len(OBSERVATION_NAMES):
            raise ValueError("V205 calibration shape mismatch")
        if inspection.shape != (calibration.shape[0], len(CONDITION_NAMES), len(OBSERVATION_NAMES)):
            raise ValueError("V205 inspection shape mismatch")
        if not np.allclose(calibration.sum(axis=-1), 1.0, atol=1e-12, rtol=0.0):
            raise ValueError("V205 calibration channel is not normalized")
        if not np.allclose(inspection.sum(axis=-1), 1.0, atol=1e-12, rtol=0.0):
            raise ValueError("V205 inspection channel is not normalized")
        if np.any(calibration <= 0.0) or np.any(inspection <= 0.0):
            raise ValueError("V205 sensing channels lack common positive support")
        if not all(np.isfinite(value).all() for value in (calibration, inspection)):
            raise ValueError("V205 sensing channel contains nonfinite values")
        calibration.setflags(write=False)
        inspection.setflags(write=False)
        object.__setattr__(self, "calibration", calibration)
        object.__setattr__(self, "inspection", inspection)


def _logical_channel(logical: str, latent: int, config: dict[str, Any]) -> np.ndarray:
    channel = config["channel"]
    if latent == LATENT_NAMES.index("OUTSIDE_UNKNOWN"):
        return np.asarray(channel["outsideUnknownForEitherCondition"], dtype=np.float64)
    target = logical
    if latent == LATENT_NAMES.index("REVERSED"):
        target = "B" if logical == "A" else "A"
    key = "knownCodebookLogicalA" if target == "A" else "knownCodebookLogicalB"
    return np.asarray(channel[key], dtype=np.float64)


def build_kernel(config: dict[str, Any]) -> tuple[FixedStageKernel, np.ndarray]:
    calibration = np.stack(
        [_logical_channel(config["channel"]["calibrationReferenceCondition"], latent, config) for latent in range(len(LATENT_NAMES))]
    )
    inspection = np.stack(
        [
            np.stack([_logical_channel(condition, latent, config) for condition in CONDITION_NAMES])
            for latent in range(len(LATENT_NAMES))
        ]
    )
    process = config["decisionProcess"]
    kernel = FixedStageKernel(
        calibration=calibration,
        inspection=inspection,
        sensing_cost=float(process["sensingCost"]),
        deferral_reward=float(process["safeDeferralReward"]),
        repair_immediate_reward=float(process["repairImmediateReward"]),
        correct_settlement_reward=float(process["automaticCorrectSettlementReward"]),
        wrong_settlement_reward=float(process["automaticWrongSettlementReward"]),
        discount=float(process["discount"]),
    )
    latent_prior = np.asarray(config["hypotheses"]["codebookPrior"], dtype=np.float64)
    condition_prior = np.asarray(config["hypotheses"]["conditionPrior"], dtype=np.float64)
    belief = latent_prior[:, None] * condition_prior[None, :]
    _validate_belief(kernel, belief)
    return kernel, belief


def _validate_belief(kernel: FixedStageKernel, belief: np.ndarray) -> np.ndarray:
    value = np.asarray(belief, dtype=np.float64)
    if value.shape != (kernel.calibration.shape[0], len(CONDITION_NAMES)):
        raise ValueError("V205 belief shape mismatch")
    if np.any(value < 0.0) or not np.isclose(value.sum(), 1.0, atol=1e-12, rtol=0.0):
        raise ValueError("V205 belief is invalid")
    return value


def sensing_step(kernel: FixedStageKernel, belief: np.ndarray, action_name: str) -> dict[str, Any]:
    value = _validate_belief(kernel, belief)
    if action_name == "calibrate":
        likelihood = np.broadcast_to(kernel.calibration[:, None, :], value.shape + (len(OBSERVATION_NAMES),))
    elif action_name == "inspect":
        likelihood = kernel.inspection
    else:
        raise ValueError("V205 sensing_step requires a sensing action")
    joint = value[:, :, None] * likelihood
    probabilities = joint.sum(axis=(0, 1))
    posteriors = {
        observation: joint[:, :, observation] / float(probability)
        for observation, probability in enumerate(probabilities)
        if probability > 0.0
    }
    return {
        "immediate_reward": kernel.sensing_cost,
        "probabilities": probabilities,
        "posteriors": posteriors,
    }


def repair_return(kernel: FixedStageKernel, belief: np.ndarray, action_name: str) -> dict[str, float | bool]:
    value = _validate_belief(kernel, belief)
    if action_name not in REPAIR_ACTIONS:
        raise ValueError("V205 repair_return requires a repair action")
    condition = CONDITION_NAMES.index(action_name.removeprefix("repair_"))
    probability_correct = float(value[:, condition].sum())
    automatic_settlement = (
        probability_correct * kernel.correct_settlement_reward
        + (1.0 - probability_correct) * kernel.wrong_settlement_reward
    )
    return {
        "immediate_reward": kernel.repair_immediate_reward,
        "automatic_settlement_reward": automatic_settlement,
        "total_return": kernel.repair_immediate_reward + kernel.discount * automatic_settlement,
        "probability_correct": probability_correct,
        "mandatory_automatic_settlement": True,
    }


def allowed_actions(config: dict[str, Any], stage: int, *, allow_defer: bool = True) -> tuple[str, ...]:
    actions = tuple(config["decisionProcess"]["allowedActionsByStage"][STAGE_NAMES[stage]])
    if not allow_defer:
        actions = tuple(action for action in actions if action != "defer")
    if not actions:
        raise ValueError("V205 stage has no allowed action")
    return actions


def next_stage(action_name: str) -> int:
    if action_name == "calibrate":
        return STAGE_NAMES.index("POST_CALIBRATION")
    if action_name == "inspect":
        return STAGE_NAMES.index("POST_INSPECTION")
    raise ValueError("V205 terminal action has no next stage")


def _select(values: Sequence[float], tolerance: float) -> tuple[int, tuple[int, ...], float]:
    maximum = max(float(value) for value in values)
    optimal = tuple(index for index, value in enumerate(values) if maximum - float(value) <= tolerance)
    return optimal[0], optimal, maximum


def plan_exact(
    kernel: FixedStageKernel,
    belief: np.ndarray,
    config: dict[str, Any],
    stage: int = 0,
    *,
    allow_defer: bool = True,
    tolerance: float | None = None,
    stats: dict[str, int] | None = None,
) -> dict[str, Any]:
    value = _validate_belief(kernel, belief)
    if stage not in range(len(STAGE_NAMES)):
        raise ValueError("V205 invalid decision stage")
    if stats is not None:
        stats["decision_nodes"] = stats.get("decision_nodes", 0) + 1
        stats["belief_checks"] = stats.get("belief_checks", 0) + 1
        stats["normalized_beliefs"] = stats.get("normalized_beliefs", 0) + int(np.isclose(value.sum(), 1.0))
    tie_tolerance = float(config["decisionProcess"]["tieTolerance"] if tolerance is None else tolerance)
    rows: list[dict[str, Any]] = []
    for action_name in allowed_actions(config, stage, allow_defer=allow_defer):
        if action_name in SENSING_ACTIONS:
            step = sensing_step(kernel, value, action_name)
            branches = {}
            continuation = 0.0
            for observation, posterior in step["posteriors"].items():
                child = plan_exact(
                    kernel,
                    posterior,
                    config,
                    next_stage(action_name),
                    allow_defer=allow_defer,
                    tolerance=tie_tolerance,
                    stats=stats,
                )
                branches[observation] = child
                continuation += float(step["probabilities"][observation]) * float(child["value"])
            rows.append(
                {
                    "action": action_name,
                    "value": float(step["immediate_reward"] + kernel.discount * continuation),
                    "immediate_reward": float(step["immediate_reward"]),
                    "branches": branches,
                    "terminal_reason": None,
                }
            )
        elif action_name in REPAIR_ACTIONS:
            repair = repair_return(kernel, value, action_name)
            rows.append(
                {
                    "action": action_name,
                    "value": float(repair["total_return"]),
                    "immediate_reward": float(repair["immediate_reward"]),
                    "automatic_settlement_reward": float(repair["automatic_settlement_reward"]),
                    "branches": {},
                    "terminal_reason": "automatic_settlement",
                }
            )
        elif action_name == "defer":
            rows.append(
                {
                    "action": action_name,
                    "value": kernel.deferral_reward,
                    "immediate_reward": kernel.deferral_reward,
                    "branches": {},
                    "terminal_reason": "safe_deferral",
                }
            )
        else:
            raise ValueError(f"V205 unknown action {action_name}")
    selected_offset, optimal_offsets, maximum = _select([row["value"] for row in rows], tie_tolerance)
    selected = rows[selected_offset]
    return {
        "stage": STAGE_NAMES[stage],
        "value": float(maximum),
        "selected_action": selected["action"],
        "optimal_actions": tuple(rows[index]["action"] for index in optimal_offsets),
        "q_values": {row["action"]: float(row["value"]) for row in rows},
        "branches": selected["branches"],
        "terminal_reason": selected["terminal_reason"],
    }


def evaluate_policy(
    kernel: FixedStageKernel,
    belief: np.ndarray,
    policy: dict[str, Any],
    stage: int = 0,
) -> float:
    value = _validate_belief(kernel, belief)
    action_name = policy["selected_action"]
    if action_name in SENSING_ACTIONS:
        step = sensing_step(kernel, value, action_name)
        continuation = sum(
            float(step["probabilities"][observation])
            * evaluate_policy(kernel, posterior, policy["branches"][observation], next_stage(action_name))
            for observation, posterior in step["posteriors"].items()
        )
        return float(step["immediate_reward"] + kernel.discount * continuation)
    if action_name in REPAIR_ACTIONS:
        return float(repair_return(kernel, value, action_name)["total_return"])
    if action_name == "defer":
        return kernel.deferral_reward
    raise ValueError(f"V205 invalid policy action {action_name}")


def point_policy(
    kernel: FixedStageKernel,
    state_belief: np.ndarray,
    latent: int,
    config: dict[str, Any],
) -> dict[str, Any]:
    point_kernel = FixedStageKernel(
        calibration=kernel.calibration[latent : latent + 1].copy(),
        inspection=kernel.inspection[latent : latent + 1].copy(),
        sensing_cost=kernel.sensing_cost,
        deferral_reward=kernel.deferral_reward,
        repair_immediate_reward=kernel.repair_immediate_reward,
        correct_settlement_reward=kernel.correct_settlement_reward,
        wrong_settlement_reward=kernel.wrong_settlement_reward,
        discount=kernel.discount,
    )
    return plan_exact(point_kernel, np.asarray(state_belief, dtype=np.float64)[None, :], config)


def _complete_open_loop_programs(config: dict[str, Any], stage: int = 0, prefix: tuple[str, ...] = ()) -> list[tuple[str, ...]]:
    programs: list[tuple[str, ...]] = []
    for action_name in allowed_actions(config, stage):
        candidate = prefix + (action_name,)
        if action_name in SENSING_ACTIONS:
            programs.extend(_complete_open_loop_programs(config, next_stage(action_name), candidate))
        else:
            programs.append(candidate)
    return programs


def evaluate_open_loop_program(
    kernel: FixedStageKernel,
    belief: np.ndarray,
    program: Sequence[str],
    stage: int = 0,
) -> float:
    if not program:
        return kernel.deferral_reward
    action_name = program[0]
    if action_name in SENSING_ACTIONS:
        step = sensing_step(kernel, belief, action_name)
        continuation = sum(
            float(step["probabilities"][observation])
            * evaluate_open_loop_program(kernel, posterior, program[1:], next_stage(action_name))
            for observation, posterior in step["posteriors"].items()
        )
        return float(step["immediate_reward"] + kernel.discount * continuation)
    if action_name in REPAIR_ACTIONS:
        return float(repair_return(kernel, belief, action_name)["total_return"])
    if action_name == "defer":
        return kernel.deferral_reward
    raise ValueError(f"V205 invalid open-loop action {action_name}")


def best_open_loop(kernel: FixedStageKernel, belief: np.ndarray, config: dict[str, Any]) -> dict[str, Any]:
    programs = _complete_open_loop_programs(config)
    rows = [(program, evaluate_open_loop_program(kernel, belief, program)) for program in programs]
    tolerance = float(config["decisionProcess"]["tieTolerance"])
    selected, optimal, maximum = _select([value for _, value in rows], tolerance)
    return {
        "value": float(maximum),
        "selected_program": rows[selected][0],
        "optimal_programs": [rows[index][0] for index in optimal],
        "program_count": len(rows),
    }


def myopic_policy(kernel: FixedStageKernel, belief: np.ndarray, config: dict[str, Any], stage: int = 0) -> dict[str, Any]:
    value = _validate_belief(kernel, belief)
    actions = allowed_actions(config, stage)
    immediate_values = [
        kernel.sensing_cost if action in SENSING_ACTIONS
        else kernel.repair_immediate_reward if action in REPAIR_ACTIONS
        else kernel.deferral_reward
        for action in actions
    ]
    selected, optimal, maximum = _select(immediate_values, float(config["decisionProcess"]["tieTolerance"]))
    action_name = actions[selected]
    branches = {}
    if action_name in SENSING_ACTIONS:
        step = sensing_step(kernel, value, action_name)
        branches = {
            observation: myopic_policy(kernel, posterior, config, next_stage(action_name))
            for observation, posterior in step["posteriors"].items()
        }
    return {
        "stage": STAGE_NAMES[stage],
        "value": float(maximum),
        "selected_action": action_name,
        "optimal_actions": tuple(actions[index] for index in optimal),
        "branches": branches,
        "terminal_reason": "automatic_settlement" if action_name in REPAIR_ACTIONS else "safe_deferral" if action_name == "defer" else None,
    }


def _mutual_information(joint: np.ndarray) -> float:
    value = np.asarray(joint, dtype=np.float64)
    row = value.sum(axis=1, keepdims=True)
    column = value.sum(axis=0, keepdims=True)
    independent = row @ column
    mask = value > 0.0
    return float(np.sum(value[mask] * np.log(value[mask] / independent[mask])))


def structural_diagnostics(kernel: FixedStageKernel, belief: np.ndarray, config: dict[str, Any]) -> dict[str, Any]:
    latent_prior = belief.sum(axis=1)
    joint = latent_prior[:, None] * kernel.calibration
    known_a = np.asarray(config["channel"]["knownCodebookLogicalA"], dtype=np.float64)
    known_b = np.asarray(config["channel"]["knownCodebookLogicalB"], dtype=np.float64)
    immediate_values = [kernel.sensing_cost, kernel.sensing_cost, kernel.repair_immediate_reward, kernel.repair_immediate_reward]
    final_allowed = allowed_actions(config, STAGE_NAMES.index("POST_INSPECTION"))
    return {
        "calibration_mutual_information_nats": _mutual_information(joint),
        "known_codebook_inspection_total_variation": float(0.5 * np.abs(known_a - known_b).sum()),
        "point_model_common_support": bool(np.all(kernel.calibration > 0.0) and np.all(kernel.inspection > 0.0)),
        "minimum_observation_probability_on_sensing_support": float(min(kernel.calibration.min(), kernel.inspection.min())),
        "maximum_initial_sensing_or_repair_immediate_reward": float(max(immediate_values)),
        "repair_immediate_rewards": {action: kernel.repair_immediate_reward for action in REPAIR_ACTIONS},
        "mandatory_automatic_settlement_rate": 1.0,
        "unfinished_sensing_safe_deferral_rate": float(
            config["decisionProcess"]["unfinishedSensingAlwaysTerminatesBySafeDeferral"]
            and config["decisionProcess"]["unfinishedSensingTerminalValue"] == kernel.deferral_reward
        ),
        "unsettled_repair_terminal_count": 0,
        "horizon_escape_path_count": int(any(action in SENSING_ACTIONS for action in final_allowed)),
        "final_stage_allowed_actions": list(final_allowed),
        "fallback_count": 0,
    }


def _policy_terminal_audit(policy: dict[str, Any]) -> dict[str, int]:
    counts = {"terminal_paths": 0, "automatic_settlement_paths": 0, "safe_deferral_paths": 0, "unsettled_paths": 0}
    stack = [policy]
    while stack:
        node = stack.pop()
        action_name = node["selected_action"]
        if action_name in REPAIR_ACTIONS:
            counts["terminal_paths"] += 1
            counts["automatic_settlement_paths"] += 1
        elif action_name == "defer":
            counts["terminal_paths"] += 1
            counts["safe_deferral_paths"] += 1
        elif action_name in SENSING_ACTIONS and node.get("branches"):
            stack.extend(node["branches"].values())
        else:
            counts["terminal_paths"] += 1
            counts["unsettled_paths"] += 1
    return counts


def _reachable_selected_actions(policy: dict[str, Any]) -> tuple[set[str], int]:
    actions: set[str] = set()
    defer_count = 0
    stack = [policy]
    while stack:
        node = stack.pop()
        action_name = node["selected_action"]
        actions.add(action_name)
        defer_count += action_name == "defer"
        stack.extend(node.get("branches", {}).values())
    return actions, defer_count


def _branch_action(policy: dict[str, Any], observation_name: str) -> str | None:
    observation = OBSERVATION_NAMES.index(observation_name)
    branch = policy.get("branches", {}).get(observation)
    return None if branch is None else branch["selected_action"]


def evaluate_oracle(config: dict[str, Any]) -> dict[str, Any]:
    kernel, belief = build_kernel(config)
    stats: dict[str, int] = {}
    exact = plan_exact(kernel, belief, config, stats=stats)
    forced = plan_exact(kernel, belief, config, allow_defer=False)

    closed_kernel = FixedStageKernel(
        calibration=kernel.calibration[:2].copy(),
        inspection=kernel.inspection[:2].copy(),
        sensing_cost=kernel.sensing_cost,
        deferral_reward=kernel.deferral_reward,
        repair_immediate_reward=kernel.repair_immediate_reward,
        correct_settlement_reward=kernel.correct_settlement_reward,
        wrong_settlement_reward=kernel.wrong_settlement_reward,
        discount=kernel.discount,
    )
    closed_belief = belief[:2].copy()
    closed_belief /= closed_belief.sum()
    closed_policy = plan_exact(closed_kernel, closed_belief, config)
    closed_value = evaluate_policy(kernel, belief, closed_policy)

    latent_masses = belief.sum(axis=1)
    map_latent = int(np.argmax(latent_masses))
    map_policy = point_policy(kernel, belief[map_latent] / latent_masses[map_latent], map_latent, config)
    map_value = evaluate_policy(kernel, belief, map_policy)
    sampled_rows = []
    sampled_value = 0.0
    for latent, mass in enumerate(latent_masses):
        policy = point_policy(kernel, belief[latent] / mass, latent, config)
        true_value = evaluate_policy(kernel, belief, policy)
        sampled_value += float(mass) * true_value
        sampled_rows.append(
            {
                "latent": LATENT_NAMES[latent],
                "mass": float(mass),
                "root_action": policy["selected_action"],
                "true_value": true_value,
            }
        )

    open_loop = best_open_loop(kernel, belief, config)
    myopic = myopic_policy(kernel, belief, config)
    myopic_value = evaluate_policy(kernel, belief, myopic)
    exact_value = float(exact["value"])
    scale = float(
        (kernel.correct_settlement_reward - kernel.wrong_settlement_reward)
        * config["decisionProcess"]["maximumControllableDecisionCount"]
    )
    exact_actions, defer_histories = _reachable_selected_actions(exact)
    terminal_audit = _policy_terminal_audit(exact)
    post_calibration_actions = {
        action for observation in OBSERVATION_NAMES if (action := _branch_action(exact, observation)) is not None
    }
    normalization_rate = stats["normalized_beliefs"] / stats["belief_checks"]
    immediate_defer = kernel.deferral_reward
    return {
        "structural": {
            **structural_diagnostics(kernel, belief, config),
            "belief_normalization_rate": normalization_rate,
            "exact_policy_terminal_audit": terminal_audit,
        },
        "return_scale": scale,
        "decision_nodes": stats["decision_nodes"],
        "exact": {
            "value": exact_value,
            "root_action": exact["selected_action"],
            "root_optimal_actions": list(exact["optimal_actions"]),
            "root_q_values": exact["q_values"],
            "action_after_root_red": _branch_action(exact, "red"),
            "action_after_root_blue": _branch_action(exact, "blue"),
            "action_after_root_green": _branch_action(exact, "green"),
            "distinct_post_calibration_selected_actions": sorted(post_calibration_actions),
            "reachable_selected_actions": sorted(exact_actions),
            "reachable_defer_history_count": defer_histories,
            "distinct_reachable_repair_actions": sorted(action for action in exact_actions if action in REPAIR_ACTIONS),
            "normalized_advantage_over_immediate_defer": (exact_value - immediate_defer) / scale,
        },
        "closed_world": {
            "true_environment_value": closed_value,
            "root_action": closed_policy["selected_action"],
            "action_after_root_green": _branch_action(closed_policy, "green"),
            "normalized_regret": (exact_value - closed_value) / scale,
        },
        "forced_commit": {
            "true_environment_value": float(forced["value"]),
            "root_action": forced["selected_action"],
            "normalized_regret": (exact_value - float(forced["value"])) / scale,
        },
        "map": {
            "latent": LATENT_NAMES[map_latent],
            "true_environment_value": map_value,
            "root_action": map_policy["selected_action"],
            "normalized_regret": (exact_value - map_value) / scale,
        },
        "posterior_sampling": {
            "true_environment_value": sampled_value,
            "normalized_regret": (exact_value - sampled_value) / scale,
            "models": sampled_rows,
        },
        "open_loop": {
            "value": open_loop["value"],
            "selected_program": list(open_loop["selected_program"]),
            "optimal_programs": [list(program) for program in open_loop["optimal_programs"]],
            "normalized_exact_advantage": (exact_value - open_loop["value"]) / scale,
            "program_count": open_loop["program_count"],
        },
        "myopic": {
            "true_environment_value": myopic_value,
            "root_action": myopic["selected_action"],
            "normalized_regret": (exact_value - myopic_value) / scale,
        },
        "immediate_defer": {"value": immediate_defer},
        "access": {
            "oracle_evaluation_count": 1,
            "language_record_read_count": 0,
            "raw_model_response_read_count": 0,
            "protected_access_count": 0,
            "model_load_count": 0,
            "model_generation_count": 0,
            "API_call_count": 0,
            "training_run_count": 0,
            "ontology_registration_count": 0,
            "trusted_state_mutation_count": 0,
            "service_call_count": 0,
            "external_side_effect_count": 0,
            "actual_execution_count": 0,
        },
    }


def audit_oracle(result: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    gates = config["oracleGates"]
    structural = result["structural"]
    terminal = structural["exact_policy_terminal_audit"]
    checks = {
        "latent_and_common_support": bool(
            len(LATENT_NAMES) == gates["requiredLatentCount"]
            and structural["point_model_common_support"] == gates["requiredPointModelCommonSupport"]
            and structural["minimum_observation_probability_on_sensing_support"]
            >= gates["minimumObservationProbabilityOnSensingSupport"] - 1e-12
        ),
        "sensing_channels_are_informative": bool(
            structural["calibration_mutual_information_nats"] >= gates["minimumCalibrationMutualInformationNats"]
            and structural["known_codebook_inspection_total_variation"] >= gates["minimumKnownCodebookInspectionTotalVariation"]
        ),
        "no_immediate_reward_shortcut": bool(
            structural["maximum_initial_sensing_or_repair_immediate_reward"] <= gates["maximumInitialSensingOrRepairImmediateReward"]
            and set(structural["repair_immediate_rewards"].values()) == {gates["requiredRepairImmediateReward"]}
        ),
        "terminal_accounting_is_complete": bool(
            structural["mandatory_automatic_settlement_rate"] == gates["requiredMandatoryAutomaticSettlementRate"]
            and structural["unfinished_sensing_safe_deferral_rate"] == gates["requiredUnfinishedSensingSafeDeferralRate"]
            and structural["unsettled_repair_terminal_count"] <= gates["maximumUnsettledRepairTerminalCount"]
            and structural["horizon_escape_path_count"] <= gates["maximumHorizonEscapePathCount"]
            and terminal["unsettled_paths"] == 0
            and terminal["terminal_paths"]
            == terminal["automatic_settlement_paths"] + terminal["safe_deferral_paths"]
        ),
        "exact_history_actions_match_open_world_mechanism": bool(
            result["exact"]["root_action"] == gates["requiredSelectedExactRootAction"]
            and result["exact"]["action_after_root_red"] == gates["requiredExactActionAfterRootRed"]
            and result["exact"]["action_after_root_blue"] == gates["requiredExactActionAfterRootBlue"]
            and result["exact"]["action_after_root_green"] == gates["requiredExactActionAfterRootGreen"]
            and result["closed_world"]["action_after_root_green"] == gates["requiredClosedWorldActionAfterRootGreen"]
        ),
        "history_varying_sensing_control_and_abstention_are_reachable": bool(
            len(result["exact"]["distinct_reachable_repair_actions"]) >= gates["minimumDistinctReachableRepairActions"]
            and result["exact"]["reachable_defer_history_count"] >= gates["minimumReachableDeferHistoryCount"]
            and len(result["exact"]["distinct_post_calibration_selected_actions"])
            >= gates["minimumDistinctPostCalibrationSelectedActions"]
        ),
        "exact_beats_immediate_defer": result["exact"]["normalized_advantage_over_immediate_defer"]
        >= gates["minimumNormalizedExactOverImmediateDeferAdvantage"],
        "closed_world_has_material_regret": result["closed_world"]["normalized_regret"]
        >= gates["minimumNormalizedClosedWorldRegret"],
        "forced_commit_has_material_regret": result["forced_commit"]["normalized_regret"]
        >= gates["minimumNormalizedForcedCommitRegret"],
        "MAP_has_material_regret": result["map"]["normalized_regret"] >= gates["minimumNormalizedMAPRegret"],
        "posterior_sampling_has_material_regret": result["posterior_sampling"]["normalized_regret"]
        >= gates["minimumNormalizedPosteriorSamplingRegret"],
        "adaptive_beats_open_loop": result["open_loop"]["normalized_exact_advantage"]
        >= gates["minimumNormalizedExactOverOpenLoopAdvantage"],
        "myopic_has_material_regret": result["myopic"]["normalized_regret"] >= gates["minimumNormalizedMyopicRegret"],
        "beliefs_finite_and_no_fallback": bool(
            structural["belief_normalization_rate"] >= gates["minimumBeliefNormalizationRate"]
            and structural["fallback_count"] <= gates["maximumFallbackCount"]
            and (not gates["requiredFiniteMetrics"] or all(
                math.isfinite(value)
                for value in (
                    result["exact"]["value"],
                    result["closed_world"]["true_environment_value"],
                    result["forced_commit"]["true_environment_value"],
                    result["map"]["true_environment_value"],
                    result["posterior_sampling"]["true_environment_value"],
                    result["open_loop"]["value"],
                    result["myopic"]["true_environment_value"],
                )
            ))
        ),
    }
    access_gates = config["accessGates"]
    access = result["access"]
    access_checks = {
        "oracle_evaluation_count_exact": access["oracle_evaluation_count"] == access_gates["requiredOracleEvaluationCount"],
        "forbidden_access_and_effects_zero": all(
            access[key] <= access_gates[gate]
            for key, gate in (
                ("language_record_read_count", "maximumLanguageRecordReadCount"),
                ("raw_model_response_read_count", "maximumRawModelResponseReadCount"),
                ("protected_access_count", "maximumProtectedAccessCount"),
                ("model_load_count", "maximumModelLoadCount"),
                ("model_generation_count", "maximumModelGenerationCount"),
                ("API_call_count", "maximumAPICallCount"),
                ("training_run_count", "maximumTrainingRunCount"),
                ("ontology_registration_count", "maximumOntologyRegistrationCount"),
                ("trusted_state_mutation_count", "maximumTrustedStateMutationCount"),
                ("service_call_count", "maximumServiceCallCount"),
                ("external_side_effect_count", "maximumExternalSideEffectCount"),
                ("actual_execution_count", "maximumActualExecutionCount"),
            )
        ),
    }
    return {
        "passed": all(checks.values()) and all(access_checks.values()),
        "scientific_gates_passed": all(checks.values()),
        "access_gates_passed": all(access_checks.values()),
        "checks": checks,
        "access_checks": access_checks,
        "result": result,
    }


__all__ = [
    "ACTION_NAMES",
    "CONDITION_NAMES",
    "FixedStageKernel",
    "LATENT_NAMES",
    "OBSERVATION_NAMES",
    "STAGE_NAMES",
    "allowed_actions",
    "audit_oracle",
    "build_kernel",
    "evaluate_oracle",
    "plan_exact",
    "repair_return",
    "sensing_step",
]
