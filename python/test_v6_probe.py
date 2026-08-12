import unittest

from evaluate_v6_mechanic_holdout_mlx import gate_report


class V6GateTests(unittest.TestCase):
    def lock(self):
        return {
            "calibration_gate": {"value": 0.8},
            "reference": {"v5_challenge_balanced_accuracy": 0.49},
            "gates": {
                "minimumCalibrationCanonicalBalancedAccuracy": 0.75,
                "minimumHoldoutCanonicalBalancedAccuracy": 0.70,
                "minimumHoldoutBootstrapLowerBound": 0.55,
                "minimumSurfaceBalancedAccuracy": 0.65,
                "minimumSurfacePredictionAgreement": 0.85,
                "minimumCompleteTripletAccuracy": 0.60,
                "minimumAbsoluteImprovementOverV5": 0.15,
                "minimumEvidenceDirectionalAccuracy": 0.75,
            },
        }

    def test_all_preregistered_gates_must_pass(self):
        report = gate_report(
            self.lock(),
            {"balanced_accuracy": 0.8},
            {"balanced_accuracy_95_percentile_interval": [0.6, 0.9]},
            {
                "entity_renamed": {"balanced_accuracy": 0.75},
                "paraphrased": {"balanced_accuracy": 0.76},
            },
            {
                "complete_triplet_accuracy": 0.7,
                "transformations": {
                    "entity_renamed": {"prediction_agreement": 0.9},
                    "paraphrased": {"prediction_agreement": 0.91},
                },
            },
            {"directional_accuracy": 0.8},
        )
        self.assertTrue(report["passed"])
        self.assertTrue(all(check["passed"] for check in report["checks"]))

        report["checks"][0]["passed"] = False
        self.assertFalse(all(check["passed"] for check in report["checks"]))

    def test_improvement_is_measured_against_locked_v5_challenge(self):
        report = gate_report(
            self.lock(),
            {"balanced_accuracy": 0.60},
            {"balanced_accuracy_95_percentile_interval": [0.56, 0.7]},
            {
                "entity_renamed": {"balanced_accuracy": 0.7},
                "paraphrased": {"balanced_accuracy": 0.7},
            },
            {
                "complete_triplet_accuracy": 0.7,
                "transformations": {
                    "entity_renamed": {"prediction_agreement": 0.9},
                    "paraphrased": {"prediction_agreement": 0.9},
                },
            },
            {"directional_accuracy": 0.8},
        )
        improvement = next(
            check for check in report["checks"]
            if check["name"] == "absolute_improvement_over_v5_challenge"
        )
        self.assertAlmostEqual(improvement["value"], 0.11)
        self.assertFalse(improvement["passed"])
        self.assertFalse(report["passed"])


if __name__ == "__main__":
    unittest.main()
