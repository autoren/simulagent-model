#!/usr/bin/env python3
"""Audit V79 implementation without evaluating its registered policies."""
from __future__ import annotations

import hashlib
import json
from typing import Any

import numpy as np

from evaluate_v79_terminal_utility_benchmark import evaluate_gates
from test_v78_clarification_benchmark import fake_fixture, tiny_terminal_kernel
from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT
from v78_clarification_benchmark import build_fixture, structural_diagnostics
from v79_terminal_utility_planning import (
    ACTIVE_UNRESOLVED_TERMINAL_UTILITY,
    TERMINAL_STATE_TERMINAL_UTILITY,
    evaluate_policy_exact,
    plan_exact,
    terminal_utility,
)


def payload_hash(value: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def main() -> None:
    design_path = PROJECT_ROOT / "configs/v79-terminal-utility-design-lock.json"
    resource_path = PROJECT_ROOT / "configs/v79-terminal-utility-resource-budget.json"
    planner_path = PROJECT_ROOT / "python/v79_terminal_utility_planning.py"
    evaluator_path = PROJECT_ROOT / "python/evaluate_v79_terminal_utility_benchmark.py"
    parent_reporter_path = PROJECT_ROOT / "python/evaluate_v78_clarification_benchmark.py"
    harness_path = PROJECT_ROOT / "python/locked_census_harness.py"
    tests_path = PROJECT_ROOT / "python/test_v79_terminal_utility_benchmark.py"
    auditor_path = (
        PROJECT_ROOT / "python/audit_and_freeze_v79_terminal_utility_implementation.py"
    )
    audit_path = PROJECT_ROOT / "outputs/v79-structured-llm-interface/implementation-audit.json"
    lock_path = PROJECT_ROOT / "configs/v79-terminal-utility-implementation-lock.json"
    outcome_dir = PROJECT_ROOT / "outputs/v79-structured-llm-interface/model-free-evaluation"
    if audit_path.exists() or lock_path.exists():
        raise RuntimeError("V79 implementation is already frozen")
    if outcome_dir.exists():
        raise RuntimeError("V79 outcome exists before implementation lock")

    design = json.loads(design_path.read_text())
    design_payload = {
        key: value for key, value in design.items() if key != "lock_payload_sha256"
    }
    config = design["resolved_config_payload"]
    resource = json.loads(resource_path.read_text())
    authorization = bool(
        payload_hash(design_payload) == design["lock_payload_sha256"]
        and design["authorization"][
            "implement_and_structurally_audit_model_free_terminal_utility"
        ]
        and not design["authorization"]["compute_V79_policy_values_or_optimal_actions"]
        and not design["authorization"]["access_local_or_API_model"]
        and not design["authorization"]["access_human_records_or_real_tools"]
    )

    fixtures = [build_fixture(config, row["name"]) for row in config["fixtures"]]
    diagnostics = [structural_diagnostics(fixture) for fixture in fixtures]
    inherited_population = bool(
        [fixture.name for fixture in fixtures]
        == [row["name"] for row in config["fixtures"]]
        and all(
            np.array_equal(fixture.initial_belief[:, 0], row["prior"])
            for fixture, row in zip(fixtures, config["fixtures"], strict=True)
        )
        and all(row["transition_normalization_rate"] == 1.0 for row in diagnostics)
        and all(row["observation_normalization_rate"] == 1.0 for row in diagnostics)
        and all(row["identical_hypothesis_support_rate"] == 1.0 for row in diagnostics)
    )

    tiny_kernel, tiny_belief = tiny_terminal_kernel()
    tiny_terminal = np.zeros_like(tiny_belief)
    tiny_terminal[:, 1] = tiny_belief[:, 0]
    tiny_policy = plan_exact(tiny_kernel, tiny_belief, 1)
    terminal_semantics = bool(
        ACTIVE_UNRESOLVED_TERMINAL_UTILITY
        == config["terminalUtility"]["activeUnresolvedAtHorizonExhaustion"]
        == -6.0
        and TERMINAL_STATE_TERMINAL_UTILITY
        == config["terminalUtility"]["terminalStateAtHorizonExhaustion"]
        == 0.0
        and terminal_utility(tiny_kernel, tiny_belief) == -6.0
        and terminal_utility(tiny_kernel, tiny_terminal) == 0.0
        and tiny_kernel.action_names[tiny_policy["selected_action"]] == "safe_preview"
        and evaluate_policy_exact(tiny_kernel, tiny_belief, tiny_policy, 1)
        == tiny_policy["value"]
        == 2.0
    )

    fake = {row["name"]: fake_fixture(row["name"]) for row in config["fixtures"]}
    fake["clear_tool_intent"]["exact"]["root_action"] = "execute_schedule_chen"
    fake["unknown_heavy_tool_intent"]["exact"]["root_action"] = "ask_full_details"
    fake["dominant_safe_preview"]["exact"]["root_action"] = "safe_preview"
    fake["dominant_safe_preview"]["map"]["normalized_regret"] = 0.0
    fake["dominant_safe_preview"]["posterior_sampling"]["normalized_regret"] = 0.0
    for row in fake.values():
        row["terminal_utility"] = {
            "initial_active_belief_at_horizon_zero": -6.0,
            "matched_terminal_belief_at_horizon_zero": 0.0,
            "exact_policy_replay_agrees": True,
        }
    zero_access = {
        "model_forward_pass_count": 0,
        "API_call_count": 0,
        "adapter_training_run_count": 0,
        "human_record_access_count": 0,
        "real_tool_call_count": 0,
        "external_side_effect_count": 0,
    }
    fake_gates = evaluate_gates(fake, config, zero_access)
    full_path = bool(
        fake_gates
        and all(isinstance(value, bool) and value for value in fake_gates.values())
    )

    action_count = resource["registeredActionCount"]
    observation_count = resource["registeredObservationCount"]
    horizon = resource["registeredHorizon"]
    conservative_nodes = sum(
        (action_count * observation_count) ** depth
        for depth in range(horizon + 1)
    )
    resources = {
        "fixture_count": len(fixtures) == resource["registeredFixtureCount"],
        "dense_kernel_bytes": all(
            row["dense_kernel_bytes"] <= resource["maximumDenseKernelBytesPerFixture"]
            for row in diagnostics
        ),
        "terminal_leaf_aware_Bellman_bound": conservative_nodes
        <= resource["maximumConservativeBellmanNodeBoundPerFixture"],
        "open_loop_sequence_count": action_count**horizon
        == resource["requiredOpenLoopSequenceCountPerFixture"],
        "durable_raw_fixture_count": len(fixtures)
        == resource["minimumDurableRawFixtureArtifactCountOnCompletedCensus"],
        "zero_external_access_budget": all(
            resource[key] == 0
            for key in (
                "maximumModelForwardPassCount",
                "maximumAPICallCount",
                "maximumAdapterTrainingRunCount",
                "maximumHumanRecordAccessCount",
                "maximumRealToolCallCount",
                "maximumExternalSideEffectCount",
            )
        ),
    }
    checks = {
        "design_lock_authorizes_model_free_implementation": authorization,
        "V78_population_and_stochastic_kernel_inherited": inherited_population,
        "terminal_utility_semantics_and_horizon_one_resolution": terminal_semantics,
        "outcome_blind_augmented_gate_full_path_smoke": full_path,
        "resource_durability_and_access_bounds_pass": all(resources.values()),
        "zero_registered_V79_policy_values_or_actions_computed": True,
        "zero_model_API_human_or_real_tool_access": True,
    }
    passed = all(checks.values())
    audit = {
        "schema_version": "79-terminal-utility-implementation-audit",
        "experiment": "v79_model_free_terminal_utility_implementation_audit",
        "passed": passed,
        "decision": (
            "freeze_implementation_and_authorize_one_durable_model_free_census"
            if passed
            else "reject_or_defer_V79_implementation"
        ),
        "checks": checks,
        "resource_checks": resources,
        "conservative_Bellman_node_bound_per_fixture": conservative_nodes,
        "access": {
            "V79_policy_value_count": 0,
            "V79_optimal_action_count": 0,
            **zero_access,
        },
    }
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    if not passed:
        print(json.dumps(audit, indent=2, sort_keys=True))
        raise SystemExit(1)

    lock = {
        "schema_version": "79-terminal-utility-implementation-lock",
        "experiment": "v79_model_free_terminal_utility_implementation_lock",
        "design_lock": str(design_path.relative_to(PROJECT_ROOT)),
        "design_lock_sha256": file_sha256(design_path),
        "resolved_config_payload": config,
        "resource_budget": str(resource_path.relative_to(PROJECT_ROOT)),
        "resource_budget_sha256": file_sha256(resource_path),
        "terminal_planner": str(planner_path.relative_to(PROJECT_ROOT)),
        "terminal_planner_sha256": file_sha256(planner_path),
        "evaluator": str(evaluator_path.relative_to(PROJECT_ROOT)),
        "evaluator_sha256": file_sha256(evaluator_path),
        "parent_reporter": str(parent_reporter_path.relative_to(PROJECT_ROOT)),
        "parent_reporter_sha256": file_sha256(parent_reporter_path),
        "census_harness": str(harness_path.relative_to(PROJECT_ROOT)),
        "census_harness_sha256": file_sha256(harness_path),
        "tests": str(tests_path.relative_to(PROJECT_ROOT)),
        "tests_sha256": file_sha256(tests_path),
        "implementation_auditor": str(auditor_path.relative_to(PROJECT_ROOT)),
        "implementation_auditor_sha256": file_sha256(auditor_path),
        "implementation_audit": str(audit_path.relative_to(PROJECT_ROOT)),
        "implementation_audit_sha256": file_sha256(audit_path),
        "authorization": {
            "modify_inherited_design_or_terminal_utility": False,
            "modify_frozen_planner_evaluator_reporter_or_harness": False,
            "run_model_free_census_once": True,
            "run_model_free_census_more_than_once": False,
            "access_local_or_API_model": False,
            "access_human_records_or_real_tools": False,
            "perform_external_side_effect": False,
        },
    }
    lock["lock_payload_sha256"] = payload_hash(lock)
    lock_path.write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n")
    print(json.dumps(audit, indent=2, sort_keys=True))
    print(
        json.dumps(
            {"lock": str(lock_path), "sha256": file_sha256(lock_path)},
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
