#!/usr/bin/env python3
"""Structurally audit V77 and freeze its one model-free evaluator attempt."""
from __future__ import annotations

import hashlib
import json
from typing import Any

import numpy as np

from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT
from v77_clarification_benchmark import (
    ACTION_NAMES,
    HYPOTHESIS_NAMES,
    NONE_HYPOTHESIS,
    OBSERVATION_NAMES,
    SEND_ACTIONS,
    ClarificationKernel,
    build_fixture,
    certified_actions,
    structural_diagnostics,
    validate_belief,
)


def payload_hash(value: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _rejects_kernel_mutation(kernel: ClarificationKernel, field: str) -> bool:
    transition = kernel.transition.copy()
    observation = kernel.observation.copy()
    reward = kernel.reward.copy()
    if field == "transition_normalization":
        transition[0, 0, 0] += 0.1
    elif field == "observation_normalization":
        observation[0, 0, 0, 0] += 0.1
    elif field == "support_mismatch":
        observation[1, 0, 0, 0] = 0.0
        observation[1, 0, 0, 1] += float(kernel.observation[1, 0, 0, 0])
    else:
        raise ValueError(field)
    try:
        ClarificationKernel(
            hypothesis_names=kernel.hypothesis_names,
            action_names=kernel.action_names,
            observation_names=kernel.observation_names,
            state_names=kernel.state_names,
            transition=transition,
            observation=observation,
            reward=reward,
            discount=kernel.discount,
            send_minimum_matching_posterior=kernel.send_minimum_matching_posterior,
            send_maximum_none_posterior=kernel.send_maximum_none_posterior,
            send_action_to_hypothesis=kernel.send_action_to_hypothesis,
            none_hypothesis=kernel.none_hypothesis,
            always_certified_actions=kernel.always_certified_actions,
        )
    except ValueError:
        return True
    return False


def main() -> None:
    design_path = PROJECT_ROOT / "configs/v77-clarification-design-lock.json"
    resource_path = PROJECT_ROOT / "configs/v77-clarification-resource-budget.json"
    core_path = PROJECT_ROOT / "python/v77_clarification_benchmark.py"
    evaluator_path = PROJECT_ROOT / "python/evaluate_v77_clarification_benchmark.py"
    tests_path = PROJECT_ROOT / "python/test_v77_clarification_benchmark.py"
    auditor_path = PROJECT_ROOT / "python/audit_and_freeze_v77_clarification_implementation.py"
    audit_path = PROJECT_ROOT / "outputs/v77-structured-llm-interface/implementation-audit.json"
    lock_path = PROJECT_ROOT / "configs/v77-clarification-implementation-lock.json"
    outcome_dir = PROJECT_ROOT / "outputs/v77-structured-llm-interface/model-free-evaluation"
    if lock_path.exists():
        raise RuntimeError("V77 implementation is already frozen")
    if outcome_dir.exists():
        raise RuntimeError("V77 outcome exists before implementation lock")

    design = json.loads(design_path.read_text())
    design_payload = {key: value for key, value in design.items() if key != "lock_payload_sha256"}
    resource = json.loads(resource_path.read_text())
    config = design["config_payload"]
    errors: list[str] = []
    design_valid = bool(
        payload_hash(design_payload) == design["lock_payload_sha256"]
        and design["authorization"]["implement_and_structurally_audit_model_free_benchmark"]
        and not design["authorization"]["compute_planner_policy_values_or_optimal_actions"]
        and not design["authorization"]["access_local_model"]
        and not design["authorization"]["access_API_model"]
    )
    if not design_valid:
        errors.append("V77 design lock or model-free authorization is invalid")

    fixtures = [build_fixture(config, row["name"]) for row in config["fixtures"]]
    diagnostics = [structural_diagnostics(fixture) for fixture in fixtures]
    exact_population = bool(
        [fixture.name for fixture in fixtures]
        == [row["name"] for row in config["fixtures"]]
        and all(fixture.kernel.hypothesis_names == HYPOTHESIS_NAMES for fixture in fixtures)
        and all(fixture.kernel.action_names == ACTION_NAMES for fixture in fixtures)
        and all(fixture.kernel.observation_names == OBSERVATION_NAMES for fixture in fixtures)
        and all(np.array_equal(fixture.initial_belief[:, 0], row["prior"])
                for fixture, row in zip(fixtures, config["fixtures"], strict=True))
    )
    if not exact_population:
        errors.append("V77 implementation does not reproduce the registered fixture census")

    structural = bool(
        all(row["observation_normalization_rate"] == 1.0 for row in diagnostics)
        and all(row["identical_hypothesis_support_rate"] == 1.0 for row in diagnostics)
        and all(row["belief_normalizes"] for row in diagnostics)
        and all(row["off_support_fallback_count"] == 0 for row in diagnostics)
        and fixtures[0].kernel.observation[
            NONE_HYPOTHESIS,
            ACTION_NAMES.index("ask_full_details"),
            0,
            OBSERVATION_NAMES.index("full_other"),
        ]
        == config["sharedParameters"]["fullQuestionReliability"]
    )
    if not structural:
        errors.append("V77 observation, belief, support, or none-hypothesis structure failed")

    ambiguous_actions = certified_actions(fixtures[0].kernel, fixtures[0].initial_belief)
    clear_actions = certified_actions(fixtures[1].kernel, fixtures[1].initial_belief)
    certification = bool(
        all(action not in ambiguous_actions for action in SEND_ACTIONS)
        and ACTION_NAMES.index("send_q2_lee") in clear_actions
        and all(
            ACTION_NAMES.index(name) not in clear_actions
            for name in ("send_annual_lee", "send_q2_patel", "send_annual_patel")
        )
    )
    if not certification:
        errors.append("V77 complete-belief irreversible-send certification failed")

    belief_mutation_detected = False
    bad_belief = fixtures[0].initial_belief.copy()
    bad_belief[0, 0] += 0.1
    try:
        validate_belief(fixtures[0].kernel, bad_belief)
    except ValueError:
        belief_mutation_detected = True
    mutation_checks = {
        "transition_normalization": _rejects_kernel_mutation(
            fixtures[0].kernel, "transition_normalization"
        ),
        "observation_normalization": _rejects_kernel_mutation(
            fixtures[0].kernel, "observation_normalization"
        ),
        "hypothesis_support_mismatch": _rejects_kernel_mutation(
            fixtures[0].kernel, "support_mismatch"
        ),
        "belief_normalization": belief_mutation_detected,
    }
    if not all(mutation_checks.values()):
        errors.append("V77 structural mutation controls were not all detected")

    action_count = resource["registeredActionCount"]
    observation_count = resource["registeredObservationCount"]
    horizon = resource["registeredHorizon"]
    conservative_nodes = sum(
        (action_count * observation_count) ** depth for depth in range(horizon)
    )
    open_loop_count = action_count**horizon
    resource_checks = {
        "fixture_count": len(fixtures) <= resource["maximumEvaluationFixtureCount"],
        "dense_kernel_bytes": all(
            row["dense_kernel_bytes"] <= resource["maximumDenseKernelBytesPerFixture"]
            for row in diagnostics
        ),
        "conservative_Bellman_nodes": conservative_nodes
        <= resource["maximumConservativeBellmanNodeBoundPerFixture"],
        "open_loop_sequences": open_loop_count
        == resource["requiredOpenLoopSequenceCountPerFixture"],
        "zero_model_API_adapter_human_and_external_access": all(
            resource[key] == 0
            for key in (
                "maximumModelForwardPassCount",
                "maximumAPICallCount",
                "maximumAdapterTrainingRunCount",
                "maximumHumanRecordAccessCount",
                "maximumExternalSideEffectCount",
            )
        ),
    }
    if not all(resource_checks.values()):
        errors.append("V77 prospective resource or access feasibility failed")

    checks = {
        "design_lock_and_model_free_authorization": design_valid,
        "exact_registered_fixture_population": exact_population,
        "normalized_shared_support_and_operational_none_state": structural,
        "complete_belief_fail_closed_send_certification": certification,
        "all_structural_mutations_detected": all(mutation_checks.values()),
        "all_resource_and_access_bounds_pass": all(resource_checks.values()),
        "zero_registered_policy_values_or_optimal_actions_computed": True,
        "zero_model_forward_pass_or_API_call_performed": True,
    }
    audit = {
        "schema_version": "77-clarification-implementation-audit",
        "experiment": "v77_model_free_implementation_audit",
        "passed": not errors and all(checks.values()),
        "decision": (
            "freeze_implementation_and_authorize_one_model_free_census"
            if not errors
            else "reject_or_defer_v77_model_free_implementation"
        ),
        "errors": errors,
        "checks": checks,
        "mutation_checks": mutation_checks,
        "resource_checks": resource_checks,
        "resources": {
            "dense_kernel_bytes_per_fixture": [
                row["dense_kernel_bytes"] for row in diagnostics
            ],
            "conservative_Bellman_node_bound_per_fixture": conservative_nodes,
            "open_loop_sequence_count_per_fixture": open_loop_count,
        },
        "access": {
            "registered_policy_value_count": 0,
            "registered_optimal_action_count": 0,
            "model_forward_pass_count": 0,
            "API_call_count": 0,
            "adapter_training_run_count": 0,
            "human_record_access_count": 0,
            "external_side_effect_count": 0,
        },
    }
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    if not audit["passed"]:
        print(json.dumps(audit, indent=2, sort_keys=True))
        raise SystemExit(1)

    lock = {
        "schema_version": "77-clarification-implementation-lock",
        "experiment": "v77_model_free_implementation_lock",
        "design_lock": str(design_path.relative_to(PROJECT_ROOT)),
        "design_lock_sha256": file_sha256(design_path),
        "design_payload": config,
        "resource_budget": str(resource_path.relative_to(PROJECT_ROOT)),
        "resource_budget_sha256": file_sha256(resource_path),
        "benchmark_core": str(core_path.relative_to(PROJECT_ROOT)),
        "benchmark_core_sha256": file_sha256(core_path),
        "evaluator": str(evaluator_path.relative_to(PROJECT_ROOT)),
        "evaluator_sha256": file_sha256(evaluator_path),
        "tests": str(tests_path.relative_to(PROJECT_ROOT)),
        "tests_sha256": file_sha256(tests_path),
        "implementation_auditor": str(auditor_path.relative_to(PROJECT_ROOT)),
        "implementation_auditor_sha256": file_sha256(auditor_path),
        "implementation_audit": str(audit_path.relative_to(PROJECT_ROOT)),
        "implementation_audit_sha256": file_sha256(audit_path),
        "authorization": {
            "modify_design_parameters_priors_gates_or_population": False,
            "modify_frozen_core_or_evaluator": False,
            "run_model_free_census_once": True,
            "run_model_free_census_more_than_once": False,
            "access_local_model": False,
            "access_API_model": False,
            "train_adapter": False,
            "collect_human_language": False,
            "perform_external_side_effect": False,
        },
    }
    lock["lock_payload_sha256"] = payload_hash(lock)
    lock_path.write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n")
    print(json.dumps(audit, indent=2, sort_keys=True))
    print(json.dumps({"lock": str(lock_path), "sha256": file_sha256(lock_path)}, indent=2))


if __name__ == "__main__":
    main()
