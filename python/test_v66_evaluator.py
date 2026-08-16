from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from evaluate_v66_reward import (
    TERMINAL_NAMES,
    aggregate_evaluation,
    atomic_write_json,
    atomic_write_jsonl,
    batched_known_model_oracle,
    duplicate_single_repeat_measure,
    failure_payload,
    reserve_attempt,
)
from v22r2_grounding import PROJECT_ROOT
from v64_external_eig import filter_public_history, load_family
from v65r3_smc2_eig import smc2_inference
from v66_bayes_adaptive_reward import (
    exact_kernel_and_belief,
    posterior_weighted_model_oracle,
)


def synthetic_implementation_audit() -> dict:
    return {
        "mutation_checks": {
            "a": True,
            "b": True,
            "allow_truth_field_in_planner_fixture": True,
        },
        "analytic_checks": {"a": True, "b": True},
        "inherited_smc_shared_stream_detected": True,
    }


def synthetic_access() -> dict[str, int]:
    return {
        "logical_evaluation_attempts": 1,
        "subset_public_records_loaded": 48,
        "V64_or_V65_evaluation_result_record_access": 0,
        "truth_field_access_count": 0,
        "candidate_omission_count": 0,
        "tie_break_violation_count": 0,
        "random_stream_collision_count": 0,
        "human_record_access_count": 0,
        "model_forward_pass_count": 0,
        "adapter_training_run_count": 0,
        "V65r3_evaluation_reruns": 0,
    }


def synthetic_rows() -> list[dict]:
    rows = []
    for index in range(48):
        rows.append(
            {
                "record_id": f"synthetic-{index}",
                "prefix_length": index // 8,
                "strategy_values": {
                    "exact_Bayes_adaptive": 1.0,
                    "pooled_SMC2_Bayes_adaptive_exact_environment": 1.0,
                    "pooled_SMC2_Bayes_adaptive_self": 1.0,
                    "posterior_weighted_model_oracle": 1.1,
                    "joint_MAP_certainty_equivalent": 1.0,
                    "persistent_posterior_sampling_mixture_32": 1.0,
                    "persistent_posterior_sampling_mixture_64": 1.0,
                    "myopic_expected_reward": 1.0,
                    "information_only_EIG": 0.0,
                    "invalid_mean_transition": 0.0,
                },
                "strategy_actions": {
                    "exact_Bayes_adaptive": 0,
                    "pooled_SMC2_Bayes_adaptive": 0,
                    "joint_MAP_certainty_equivalent": 0,
                    "myopic_expected_reward": 0,
                    "information_only_EIG": 1,
                    "invalid_mean_transition": 1,
                },
                "primary": {
                    "exact_value_regret": 0.0,
                    "strict_optimal_membership": True,
                    "epsilon_optimal_membership": True,
                    "root_Q_absolute_errors": [0.0] * 4,
                    "self_value_absolute_calibration_error": 0.0,
                    "MAP_minus_SMC2_value": 0.0,
                    "mixture32_minus_SMC2_value": 0.0,
                    "oracle_dominance_residual": 0.0,
                    "exact_Bellman_root_Q_reference_error": 0.0,
                    "exact_policy_evaluation_reference_error": 0.0,
                },
                "persistent_mixture": {
                    "root_action_distribution": [1.0, 0.0, 0.0, 0.0],
                },
                "repeat_diagnostics": [
                    {
                        "repeat": 0,
                        "selected_action": 1,
                        "exact_environment_value": 0.0,
                    }
                ],
                "exact_policy": {"optimal_actions": [0]},
                "integrity": {
                    "finite": True,
                    "normalizes": True,
                    "candidate_complete": True,
                    "tie_break_valid": True,
                    "invalid_mean_transition_labeled_invalid": True,
                },
            }
        )
    return rows


class V66EvaluatorTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = json.loads(
            (PROJECT_ROOT / "configs/v66-design-lock.json").read_text()
        )["config_payload"]

    def test_atomic_json_and_jsonl_leave_no_temporary(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            atomic_write_json(root / "value.json", {"b": 2, "a": 1})
            atomic_write_jsonl(root / "rows.jsonl", [{"x": 1}, {"x": 2}])
            self.assertEqual(json.loads((root / "value.json").read_text()), {"a": 1, "b": 2})
            self.assertEqual(2, len((root / "rows.jsonl").read_text().splitlines()))
            self.assertFalse(any(path.name.startswith(".") for path in root.iterdir()))

    def test_attempt_marker_consumes_one_shot(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            marker = {"logical_evaluation_attempt": 1}
            path = reserve_attempt(root, marker)
            self.assertEqual(marker, json.loads(path.read_text()))
            with self.assertRaises(RuntimeError):
                reserve_attempt(root, marker)

    def test_every_terminal_artifact_blocks_attempt(self) -> None:
        for name in TERMINAL_NAMES:
            with self.subTest(name=name), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                (root / name).write_text("{}\n")
                with self.assertRaises(RuntimeError):
                    reserve_attempt(root, {"logical_evaluation_attempt": 1})

    def test_failure_payload_is_terminal_non_authorizing_and_bound(self) -> None:
        lock_path = PROJECT_ROOT / "configs/v66-implementation-lock.json"
        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT) as directory:
            attempt = Path(directory) / "attempt.json"
            atomic_write_json(attempt, {"logical_evaluation_attempt": 1})
            try:
                raise ValueError("synthetic V66 durable failure")
            except ValueError as error:
                payload = failure_payload(
                    lock_path=lock_path,
                    attempt_path=attempt,
                    stage="synthetic_stage",
                    progress={"records_completed": 7},
                    access={"logical_evaluation_attempts": 1},
                    error=error,
                )
            self.assertFalse(payload["passed"])
            self.assertTrue(payload["one_shot_authorization_consumed"])
            self.assertEqual("do_not_authorize_policy_verification", payload["decision"])
            self.assertFalse(payload["claim_boundary"]["V66_rerun_authorized"])

    def test_valid_aggregate_fixture_passes_all_noncompensatory_gates(self) -> None:
        result = aggregate_evaluation(
            synthetic_rows(),
            self.config,
            synthetic_implementation_audit(),
            synthetic_access(),
        )
        self.assertTrue(result["passed"], result["failed_gates"])
        self.assertEqual(48, result["integrity"]["records"])
        self.assertGreaterEqual(result["controls"]["detected_or_dominated"], 4)

    def test_aggregate_rejects_accuracy_access_and_control_mutants(self) -> None:
        rows = synthetic_rows()
        regret = copy.deepcopy(rows)
        for row in regret:
            row["primary"]["exact_value_regret"] = 0.1
        access = synthetic_access()
        access["truth_field_access_count"] = 1
        controls = copy.deepcopy(rows)
        for row in controls:
            row["strategy_values"]["information_only_EIG"] = 1.0
            row["strategy_actions"]["information_only_EIG"] = 0
            row["repeat_diagnostics"][0]["exact_environment_value"] = 1.0
            row["repeat_diagnostics"][0]["selected_action"] = 0
            row["integrity"]["invalid_mean_transition_labeled_invalid"] = False
        self.assertFalse(aggregate_evaluation(regret, self.config, synthetic_implementation_audit(), synthetic_access())["passed"])
        self.assertFalse(aggregate_evaluation(rows, self.config, synthetic_implementation_audit(), access)["passed"])
        self.assertFalse(aggregate_evaluation(controls, self.config, synthetic_implementation_audit(), synthetic_access())["passed"])

    def test_batched_oracle_matches_locked_per_model_oracle(self) -> None:
        family = load_family(quadrature_nodes=3)
        exact, _ = filter_public_history(family, "left", ["n"], ["neither"])
        kernel, belief = exact_kernel_and_belief(family, exact)
        batched = batched_known_model_oracle(kernel, belief, 2)
        reference = posterior_weighted_model_oracle(kernel, belief, 2)
        self.assertAlmostEqual(batched["value"], reference["value"], places=11)
        self.assertFalse(batched["physical_state_revealed"])

    def test_single_repeat_duplication_preserves_measure(self) -> None:
        family = load_family(quadrature_nodes=3)
        record = {
            "record_id": "v66-evaluator-synthetic",
            "prefix_length": 1,
            "initial_observation": "left",
            "actions": ["n"],
            "observations": ["neither"],
        }
        smc = json.loads(
            (PROJECT_ROOT / "configs/v65r3-design-lock.json").read_text()
        )["config_payload"]
        repeat = smc2_inference(family, record, smc, 7, 0)
        duplicated = duplicate_single_repeat_measure(repeat)
        self.assertEqual(3, duplicated["repeat_count"])
        self.assertEqual(
            "one_repeat_duplicated_three_times_without_weight_change",
            duplicated["diagnostic_source"],
        )
        self.assertAlmostEqual(1.0, sum(atom["weight"] for atom in duplicated["atoms"]), places=12)

    def test_source_reserves_attempt_before_subset_read_and_serializes_both_terminals(self) -> None:
        source = (PROJECT_ROOT / "python/evaluate_v66_reward.py").read_text()
        run = source[source.index("def run_evaluation"):source.index("def main()")]
        self.assertLess(run.index("reserve_attempt("), run.index("read_jsonl("))
        self.assertIn('atomic_write_json(output_dir / "failure.json", failure)', run)
        self.assertIn('atomic_write_json(output_dir / "result.json", result)', run)


if __name__ == "__main__":
    unittest.main()
