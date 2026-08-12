import unittest

from run_v8_lomo_diagnostics import gate_report


class V8LomoGateTests(unittest.TestCase):
    def test_every_fold_and_surface_is_hard(self):
        metrics = {
            "pair_difference": {
                "canonical": {"accuracy": 0.8},
                "entity_renamed": {"accuracy": 0.59},
            },
            "pointwise_score_direction": {
                "canonical": {"accuracy": 0.8},
                "entity_renamed": {"accuracy": 0.8},
            },
        }
        folds = {"mechanic_a": metrics, "mechanic_b": metrics}
        report = gate_report(folds, {
            "minimumEveryFoldPairDifferenceAccuracy": 0.6,
            "minimumMeanPairDifferenceAccuracy": 0.7,
            "minimumEveryFoldPointwiseScoreDirection": 0.6,
        })
        self.assertFalse(report["passed"])
        self.assertEqual(report["checks"][0]["value"], 0.59)


if __name__ == "__main__":
    unittest.main()
