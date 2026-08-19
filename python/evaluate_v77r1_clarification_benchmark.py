#!/usr/bin/env python3
"""Run the one authorized V77r1 mechanical retry after V77 execution failure."""
from __future__ import annotations

import argparse
import json
from typing import Any

import evaluate_v77_clarification_benchmark as v77_evaluator
from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT
from v77_clarification_benchmark import evaluate_policy_exact
from v77r1_execution_repair import complete_terminal_branches


def _control_summary_r1(
    kernel,
    belief,
    policy: dict[str, Any],
    horizon: int,
) -> dict[str, Any]:
    repaired = complete_terminal_branches(kernel, belief, policy, horizon)
    violations: list[dict[str, Any]] = []
    value = evaluate_policy_exact(
        kernel,
        belief,
        repaired,
        horizon,
        certificate_violations=violations,
    )
    return {
        "value": float(value),
        "root_action": v77_evaluator.ACTION_NAMES[int(policy["selected_action"])],
        "complete_belief_certificate_violation_count": len(violations),
        "shadow_only": True,
        "execution_repair": "terminal_successor_branches_materialized",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--lock", default="configs/v77r1-clarification-execution-lock.json"
    )
    args = parser.parse_args()
    lock_path = PROJECT_ROOT / args.lock
    lock = json.loads(lock_path.read_text())
    payload = {key: value for key, value in lock.items() if key != "lock_payload_sha256"}
    if v77_evaluator.payload_hash(payload) != lock["lock_payload_sha256"]:
        raise RuntimeError("V77r1 execution lock payload drifted")
    if not lock["authorization"]["run_mechanical_retry_once"]:
        raise RuntimeError("V77r1 lock does not authorize the mechanical retry")
    for path_key, hash_key in (
        ("parent_implementation_lock", "parent_implementation_lock_sha256"),
        ("parent_failure", "parent_failure_sha256"),
        ("benchmark_core", "benchmark_core_sha256"),
        ("parent_evaluator", "parent_evaluator_sha256"),
        ("execution_repair", "execution_repair_sha256"),
        ("retry_evaluator", "retry_evaluator_sha256"),
    ):
        if file_sha256(PROJECT_ROOT / lock[path_key]) != lock[hash_key]:
            raise RuntimeError(f"V77r1 locked dependency drifted: {path_key}")

    output_dir = (
        PROJECT_ROOT
        / "outputs/v77-structured-llm-interface/model-free-evaluation-r1"
    )
    if output_dir.exists():
        raise RuntimeError("V77r1 model-free evaluation already exists")
    output_dir.mkdir(parents=True)
    access = {
        "attempt_number": 2,
        "execution_retry_number": 1,
        "model_forward_pass_count": 0,
        "API_call_count": 0,
        "adapter_training_run_count": 0,
        "human_record_access_count": 0,
        "external_side_effect_count": 0,
        "real_tool_call_count": 0,
        "real_message_or_file_send_count": 0,
    }
    (output_dir / "attempt.json").write_text(
        json.dumps(access, indent=2, sort_keys=True) + "\n"
    )

    # Patch only the shadow-control summary used by the frozen evaluator. The
    # exact planner, fixture population, gates, and all registered parameters
    # remain those in the parent lock.
    v77_evaluator._control_summary = _control_summary_r1
    config = lock["design_payload"]
    fixtures = {
        row["name"]: v77_evaluator.evaluate_fixture(row, config)
        for row in config["fixtures"]
    }
    gates = v77_evaluator.evaluate_gates(fixtures, config, access)
    result = {
        "schema_version": "77r1-clarification-benchmark-outcome",
        "experiment": "v77r1_model_free_clarification_benchmark_census",
        "claim_boundary": (
            "project-authored, synthetic, model-free development benchmark; "
            "mechanical retry after a recorded evaluator execution failure; "
            "not human-language, external-benchmark, model, or safety evidence"
        ),
        "parent_attempt_status": "execution_failure_without_result_artifact",
        "registered_design_changed": False,
        "exact_planner_changed": False,
        "passed": all(gates.values()),
        "decision": (
            "freeze_model_free_benchmark_and_authorize_local_model_protocol_preregistration"
            if all(gates.values())
            else "freeze_model_free_benchmark_design_failure_without_parameter_tuning"
        ),
        "gates": gates,
        "fixtures": fixtures,
        "access": access,
    }
    (output_dir / "result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
