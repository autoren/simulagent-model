import unittest

import numpy as np

from evaluate_v13_token_local import features_for_head, select_decision
from extract_v13_token_local_mlx import hypothesis_from_text


class V13TokenLocalTests(unittest.TestCase):
    def test_hypothesis_parser_requires_terminal_marker(self):
        text = "Evidence excerpt: x\nCurrent-state hypothesis: the relay is active"
        self.assertEqual(hypothesis_from_text(text), "the relay is active")
        with self.assertRaises(RuntimeError):
            hypothesis_from_text("Evidence excerpt: x")

    def test_locked_features_have_expected_pair_algebra(self):
        la = np.asarray([[3.0, 5.0]], dtype=np.float32)
        li = np.asarray([[1.0, 2.0]], dtype=np.float32)
        ma = np.asarray([[6.0, 4.0]], dtype=np.float32)
        mi = np.asarray([[2.0, 2.0]], dtype=np.float32)
        np.testing.assert_array_equal(
            features_for_head("hypothesis_last_linear", la, li, ma, mi), [[2.0, 3.0]]
        )
        np.testing.assert_array_equal(
            features_for_head("hypothesis_mean_linear", la, li, ma, mi), [[4.0, 2.0]]
        )
        np.testing.assert_array_equal(
            features_for_head("hypothesis_token_joint_mlp", la, li, ma, mi),
            [[2.0, 3.0, 4.0, 2.0, 2.0, 3.5]],
        )

    def test_decision_order_is_fixed(self):
        failed = {"gates": {"passed": False}}
        passed = {"gates": {"passed": True}}
        heads = {
            "hypothesis_last_linear": failed,
            "hypothesis_mean_linear": passed,
            "hypothesis_token_joint_mlp": passed,
        }
        self.assertEqual(
            select_decision(heads),
            "token_local_linear_relation_accessible_repair_temporal_with_hypothesis_mean",
        )
        heads["hypothesis_mean_linear"] = failed
        self.assertEqual(select_decision(heads), "token_local_nonlinear_relation_accessible_repair_temporal")
        heads["hypothesis_token_joint_mlp"] = failed
        self.assertEqual(
            select_decision(heads),
            "token_local_frozen_readout_insufficient_stop_probes_redesign_supervision",
        )


if __name__ == "__main__":
    unittest.main()
