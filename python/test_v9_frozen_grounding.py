import unittest

from evaluate_v9_frozen_grounding import gate_report, token_f1


class V9FrozenGroundingTests(unittest.TestCase):
    def test_token_f1_rewards_partial_and_exact_span_overlap(self):
        self.assertEqual(token_f1("the hatch is open", "the hatch is open"), 1.0)
        self.assertGreater(token_f1("hatch open", "the hatch is open"), 0.5)
        self.assertEqual(token_f1("generator", "mirror"), 0.0)

    def test_every_axis_and_downstream_gate_is_hard(self):
        overall = {
            "span_accuracy": 0.8,
            "allowed_values_accuracy": 0.8,
            "temporal_accuracy": 0.8,
            "symbolic_identifiability": {"balanced_accuracy": 0.8},
            "complete_flip_pair_accuracy": 0.8,
        }
        results = {
            "context": {"kind": "context", "overall": dict(overall)},
            "mechanic:a": {"kind": "mechanic", "overall": dict(overall)},
            "template:a": {"kind": "template", "overall": dict(overall)},
            "operator:a": {"kind": "operator", "overall": {**overall, "temporal_accuracy": 0.64}},
        }
        report = gate_report(results, {
            "minimumContextSpanAccuracy": 0.7,
            "minimumEveryMechanicSpanAccuracy": 0.6,
            "minimumEveryTemplateSpanAccuracy": 0.6,
            "minimumEveryOperatorSpanAccuracy": 0.6,
            "minimumEveryFoldAllowedValuesAccuracy": 0.65,
            "minimumEveryFoldTemporalAccuracy": 0.65,
            "minimumEveryFoldSymbolicBalancedAccuracy": 0.65,
            "minimumEveryFoldCompleteFlipPairAccuracy": 0.6,
        })
        self.assertFalse(report["passed"])
        self.assertEqual(
            [check["name"] for check in report["checks"] if not check["passed"]],
            ["minimum_fold_temporal_accuracy"],
        )


if __name__ == "__main__":
    unittest.main()
