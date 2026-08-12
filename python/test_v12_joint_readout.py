import unittest

import numpy as np

from evaluate_v12_joint_readout import decision, gate_report, model_features


def result(accuracy, surface=None):
    return {
        "overall": {"accuracy": accuracy},
        "by_surface": {"canonical": {"accuracy": accuracy if surface is None else surface}},
    }


class V12JointReadoutTests(unittest.TestCase):
    def test_primary_comparison_is_signed_difference(self):
        active = np.asarray([[3.0, 1.0]], dtype=np.float32)
        inactive = np.asarray([[1.0, 4.0]], dtype=np.float32)
        actual = model_features("signed_difference_linear", active, inactive)
        np.testing.assert_array_equal(actual, [[2.0, -3.0]])
        np.testing.assert_array_equal(
            model_features("signed_difference_linear", inactive, active), -actual
        )

    def test_joint_mlp_gets_mean_and_difference(self):
        active = np.asarray([[3.0, 1.0]], dtype=np.float32)
        inactive = np.asarray([[1.0, 5.0]], dtype=np.float32)
        np.testing.assert_array_equal(
            model_features("joint_mlp", active, inactive), [[2.0, 3.0, 2.0, -4.0]]
        )

    def test_gates_use_worst_fold_and_surface(self):
        folds = {
            "a": result(0.8, 0.7),
            "b": result(0.69, 0.64),
        }
        report = gate_report(folds, {
            "minimumEveryFoldAccuracy": 0.7,
            "minimumEverySurfaceAccuracy": 0.65,
        })
        self.assertFalse(report["passed"])
        self.assertEqual([round(item["value"], 2) for item in report["checks"]], [0.69, 0.64])

    def test_decision_prefers_smallest_primary_then_conditional(self):
        failed = {model: {"gates": {"passed": False}} for model in (
            "qwen35_0_8b", "qwen35_4b", "qwen35_9b"
        )}
        primary = {**failed, "qwen35_4b": {"gates": {"passed": True}}}
        self.assertEqual(
            decision(primary, None),
            "joint_linear_relation_accessible_repair_temporal_with_qwen35_4b",
        )
        conditional = {**failed, "qwen35_9b": {"gates": {"passed": True}}}
        self.assertEqual(
            decision(failed, conditional),
            "joint_nonlinear_relation_accessible_repair_temporal_with_qwen35_9b",
        )
        self.assertEqual(
            decision(failed, failed),
            "frozen_final_state_joint_readout_insufficient_extract_token_span_interactions",
        )


if __name__ == "__main__":
    unittest.main()
