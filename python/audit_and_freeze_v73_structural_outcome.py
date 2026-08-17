#!/usr/bin/env python3
"""Independently audit and freeze the negative V73 structural outcome."""
from __future__ import annotations

from itertools import product
import hashlib
import json
from typing import Any, Sequence

import numpy as np

from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT
from v73_imprl_maintenance_source import ACTION_NAMES, build_family, fixed_structural_policy


def payload_hash(value: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def independent_step(
    transition: np.ndarray,
    observation: np.ndarray,
    reward: np.ndarray,
    belief: np.ndarray,
    action: int,
) -> tuple[float, np.ndarray, dict[int, np.ndarray]]:
    predicted = np.einsum("zs,sq->zq", belief, transition[action])
    joint = predicted[:, :, None] * observation[:, action]
    probabilities = joint.sum(axis=(0, 1))
    posteriors = {
        index: joint[:, :, index] / float(probability)
        for index, probability in enumerate(probabilities)
        if probability > 0.0
    }
    immediate = float(
        np.einsum("zs,sq,sq->", belief, transition[action], reward[action])
    )
    return immediate, probabilities, posteriors


def independent_policy_value(
    transition: np.ndarray,
    observation: np.ndarray,
    reward: np.ndarray,
    discount: float,
    belief: np.ndarray,
    policy: dict[str, Any],
    horizon: int,
) -> float:
    if horizon == 0:
        return 0.0
    action = int(policy["selected_action"])
    immediate, probabilities, posteriors = independent_step(
        transition, observation, reward, belief, action
    )
    continuation = 0.0
    if horizon > 1:
        for obs, posterior in posteriors.items():
            continuation += float(probabilities[obs]) * independent_policy_value(
                transition,
                observation,
                reward,
                discount,
                posterior,
                policy["branches"][obs],
                horizon - 1,
            )
    return float(immediate + discount * continuation)


def independent_sequence_value(
    transition: np.ndarray,
    observation: np.ndarray,
    reward: np.ndarray,
    discount: float,
    belief: np.ndarray,
    actions: Sequence[int],
) -> float:
    if not actions:
        return 0.0
    immediate, probabilities, posteriors = independent_step(
        transition, observation, reward, belief, int(actions[0])
    )
    continuation = sum(
        float(probabilities[obs])
        * independent_sequence_value(
            transition,
            observation,
            reward,
            discount,
            posterior,
            actions[1:],
        )
        for obs, posterior in posteriors.items()
    )
    return float(immediate + discount * continuation)


def main() -> None:
    structural_lock_path = (
        PROJECT_ROOT / "configs/v73-active-sensing-structural-lock.json"
    )
    structural_audit_path = PROJECT_ROOT / "outputs/v73-active-sensing/structural-audit.json"
    report_path = PROJECT_ROOT / "docs/v73-active-sensing-structural-results.md"
    auditor_path = PROJECT_ROOT / "python/audit_and_freeze_v73_structural_outcome.py"
    audit_path = PROJECT_ROOT / "outputs/v73-active-sensing/structural-outcome-audit.json"
    lock_path = PROJECT_ROOT / "configs/v73-active-sensing-structural-outcome-lock.json"
    if lock_path.exists():
        raise RuntimeError("V73 structural outcome is already independently frozen")

    lock = json.loads(structural_lock_path.read_text())
    lock_payload = {
        key: value for key, value in lock.items() if key != "lock_payload_sha256"
    }
    recorded = json.loads(structural_audit_path.read_text())
    family = build_family()
    kernel = family.kernel
    policy = fixed_structural_policy(5)
    fixed_value = independent_policy_value(
        kernel.transition,
        kernel.observation,
        kernel.reward,
        kernel.discount,
        family.initial_belief,
        policy,
        5,
    )
    rows = [
        (
            actions,
            independent_sequence_value(
                kernel.transition,
                kernel.observation,
                kernel.reward,
                kernel.discount,
                family.initial_belief,
                actions,
            ),
        )
        for actions in product(range(len(ACTION_NAMES)), repeat=5)
    ]
    best_actions, best_value = max(rows, key=lambda row: row[1])
    reward_span = float(kernel.reward.max() - kernel.reward.min())
    scale = max(1.0, reward_span * sum(kernel.discount**depth for depth in range(5)))
    normalized = (fixed_value - best_value) / scale

    report = report_path.read_text()
    checks = {
        "structural_lock_payload_valid": payload_hash(lock_payload)
        == lock["lock_payload_sha256"],
        "structural_audit_hash_valid": file_sha256(structural_audit_path)
        == lock["audit_sha256"],
        "recorded_structural_result_is_negative": bool(
            not recorded["passed"]
            and recorded["failed_gates"]
            == ["fixed_adaptive_policy_beats_best_open_loop"]
        ),
        "independent_fixed_policy_value_matches": bool(
            np.isclose(
                fixed_value,
                recorded["fixed_adaptive_policy_value"],
                atol=1e-12,
                rtol=0.0,
            )
        ),
        "independent_best_open_loop_matches": bool(
            np.isclose(
                best_value,
                recorded["best_open_loop_value"],
                atol=1e-12,
                rtol=0.0,
            )
            and [ACTION_NAMES[action] for action in best_actions]
            == recorded["best_open_loop_actions"]
        ),
        "independent_normalized_advantage_matches": bool(
            np.isclose(
                normalized,
                recorded[
                    "fixed_adaptive_over_open_loop_normalized_advantage"
                ],
                atol=1e-15,
                rtol=0.0,
            )
        ),
        "frozen_threshold_failed": normalized < 0.005,
        "forbidden_planner_calls_remain_zero": all(
            recorded["access"][key] == 0
            for key in (
                "exact_Bayes_adaptive_calls",
                "MAP_calls",
                "posterior_sampling_calls",
                "myopic_calls",
            )
        ),
        "development_evaluator_does_not_exist": not (
            PROJECT_ROOT / "python/evaluate_v73_imprl_development.py"
        ).exists(),
        "report_states_negative_and_stop_boundary": all(
            phrase in report
            for phrase in (
                "useful structural negative result",
                "The branch stops before exact Bayes-adaptive",
                "0.0004026591008991246",
                "may not be tuned or reused",
            )
        ),
    }
    errors = [name for name, passed in checks.items() if not passed]
    outcome_audit = {
        "schema_version": "73-active-sensing-structural-outcome",
        "experiment": "v73_independent_structural_outcome_audit",
        "passed": not errors,
        "errors": errors,
        "checks": checks,
        "independent": {
            "fixed_adaptive_policy_value": fixed_value,
            "best_open_loop_value": best_value,
            "best_open_loop_actions": [ACTION_NAMES[action] for action in best_actions],
            "return_scale": scale,
            "normalized_advantage": normalized,
            "open_loop_sequences_recomputed": len(rows),
        },
        "access": {
            "independent_fixed_policy_values_recomputed": 1,
            "independent_open_loop_values_recomputed": len(rows),
            "optimal_contingent_planner_calls": 0,
            "protected_source_access_count": 0,
            "human_record_access_count": 0,
            "model_forward_pass_count": 0,
            "adapter_training_run_count": 0,
        },
    }
    audit_path.write_text(json.dumps(outcome_audit, indent=2, sort_keys=True) + "\n")
    if errors:
        print(json.dumps(outcome_audit, indent=2, sort_keys=True))
        raise SystemExit(1)

    outcome_lock = {
        "schema_version": "73-active-sensing-structural-outcome",
        "experiment": "v73_structural_outcome_lock",
        "structural_lock": str(structural_lock_path.relative_to(PROJECT_ROOT)),
        "structural_lock_sha256": file_sha256(structural_lock_path),
        "structural_audit": str(structural_audit_path.relative_to(PROJECT_ROOT)),
        "structural_audit_sha256": file_sha256(structural_audit_path),
        "report": str(report_path.relative_to(PROJECT_ROOT)),
        "report_sha256": file_sha256(report_path),
        "auditor": str(auditor_path.relative_to(PROJECT_ROOT)),
        "auditor_sha256": file_sha256(auditor_path),
        "outcome_audit": str(audit_path.relative_to(PROJECT_ROOT)),
        "outcome_audit_sha256": file_sha256(audit_path),
        "outcome": {
            "passed_all_structural_gates": False,
            "failed_gate_count": 1,
            "fixed_adaptive_over_open_loop_normalized_advantage": normalized,
            "exact_development_evaluation_count": 0,
            "protected_source_access_count": 0,
        },
        "authorization": {
            "modify_or_rerun_V73": False,
            "run_V73_exact_development_evaluation": False,
            "select_or_inspect_V73_confirmation_sources": False,
            "design_materially_new_successor_after_fresh_preregistration": True,
            "reuse_V73_source_component_adapter_or_parameters_for_successor_outcomes": False,
            "run_SMC2": False,
            "human_data": False,
            "model_access": False,
            "adapter_training": False,
        },
    }
    outcome_lock["lock_payload_sha256"] = payload_hash(outcome_lock)
    lock_path.write_text(json.dumps(outcome_lock, indent=2, sort_keys=True) + "\n")
    print(json.dumps(outcome_audit, indent=2, sort_keys=True))
    print(json.dumps({"lock": str(lock_path), "sha256": file_sha256(lock_path)}, indent=2))


if __name__ == "__main__":
    main()
