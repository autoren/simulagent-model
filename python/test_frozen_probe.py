import unittest

import numpy as np

from extract_frozen_qwen_features_mlx import input_variant, quartile_layers
from train_frozen_linear_probe import error_concentration, make_probe


class FrozenProbeTests(unittest.TestCase):
    def test_quartile_layers_include_final_layer(self) -> None:
        self.assertEqual(quartile_layers(24), [6, 12, 18, 24])

    def test_no_history_removes_both_history_views_without_mutating_record(self) -> None:
        record = {
            "agent_input": {
                "recent_history": [{"action": "wait"}],
                "observation": {"memories": ["waited"], "turn": 1},
            }
        }
        transformed = input_variant(record, "no_history")
        self.assertNotIn("recent_history", transformed)
        self.assertNotIn("memories", transformed["observation"])
        self.assertIn("recent_history", record["agent_input"])
        self.assertEqual(transformed["task"], "classify_identifiability")

    def test_probe_seed_is_applied_to_classifier(self) -> None:
        probe = make_probe(0.1, seed=2)
        self.assertEqual(probe.named_steps["classifier"].random_state, 2)
        self.assertEqual(probe.named_steps["classifier"].solver, "saga")

    def test_probe_preserves_float32_scores(self) -> None:
        features = np.asarray([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
        labels = np.asarray([False, False, True, True])
        probe = make_probe(0.1).fit(features, labels)
        self.assertEqual(probe.named_steps["classifier"].coef_.dtype, np.float32)
        self.assertEqual(probe.decision_function(features).dtype, np.float32)

    def test_error_concentration_counts_partial_and_complete_groups(self) -> None:
        rows = [
            {"split_group": "a", "gold_ambiguous": False, "score": 1.0},
            {"split_group": "a", "gold_ambiguous": True, "score": 1.0},
            {"split_group": "b", "gold_ambiguous": True, "score": -1.0},
        ]
        self.assertEqual(
            error_concentration(rows, 0.0),
            {
                "errors": 2,
                "context_groups_with_errors": 2,
                "context_groups": 2,
                "completely_misclassified_context_groups": 1,
            },
        )


if __name__ == "__main__":
    unittest.main()
