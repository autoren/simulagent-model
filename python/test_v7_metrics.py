import unittest

from v7_metrics import gate_report, paired_evidence_metrics, worst_stratum_metrics


class V7MetricTests(unittest.TestCase):
    def rows(self):
        return [
            {
                "surface_variant": "canonical",
                "evidence_intervention_kind": "oracle_label_change",
                "evidence_pair_id": "pair",
                "evidence_variant": "announced",
                "action_template": "use:tone",
                "split_group": "g1",
                "gold_ambiguous": False,
                "score": -1.0,
            },
            {
                "surface_variant": "canonical",
                "evidence_intervention_kind": "oracle_label_change",
                "evidence_pair_id": "pair",
                "evidence_variant": "unobservable",
                "action_template": "use:tone",
                "split_group": "g1",
                "gold_ambiguous": True,
                "score": 1.0,
            },
        ] * 2

    def test_paired_direction_requires_order_and_both_correct(self):
        result = paired_evidence_metrics(self.rows(), 0.0)
        self.assertEqual(result["groups"], 1)
        self.assertEqual(result["paired_score_directional_accuracy"], 1.0)
        self.assertEqual(result["evidence_directional_accuracy"], 1.0)

    def test_worst_stratum_uses_supported_two_label_cells(self):
        result = worst_stratum_metrics(self.rows(), 0.0, minimum_support=2)
        self.assertGreater(result["eligible_strata"], 0)
        self.assertEqual(result["worst"]["balanced_accuracy"], 1.0)

    def test_gate_report_requires_every_gate(self):
        lock = {
            "calibration_gate": {"value": 0.8},
            "gates": {
                "minimumCalibrationCanonicalBalancedAccuracy": 0.75,
                "minimumHoldoutCanonicalBalancedAccuracy": 0.70,
                "minimumHoldoutBootstrapLowerBound": 0.55,
                "minimumSurfaceBalancedAccuracy": 0.65,
                "minimumSurfacePredictionAgreement": 0.85,
                "minimumCompleteTripletAccuracy": 0.60,
                "minimumEvidenceDirectionalAccuracy": 0.75,
                "minimumPairedScoreDirectionalAccuracy": 0.75,
                "minimumWorstStratumBalancedAccuracy": 0.55,
            },
        }
        canonical = {"balanced_accuracy": 0.8}
        bootstrap = {"balanced_accuracy_95_percentile_interval": [0.6, 0.9]}
        surfaces = {
            "entity_renamed": {"balanced_accuracy": 0.8},
            "paraphrased": {"balanced_accuracy": 0.8},
        }
        invariance = {
            "complete_triplet_accuracy": 0.8,
            "transformations": {
                "entity_renamed": {"prediction_agreement": 0.9},
                "paraphrased": {"prediction_agreement": 0.9},
            },
        }
        paired = {
            "evidence_directional_accuracy": 0.8,
            "paired_score_directional_accuracy": 0.8,
        }
        worst = {"worst": {"balanced_accuracy": 0.7}}
        self.assertTrue(gate_report(lock, canonical, bootstrap, surfaces, invariance, paired, worst)["passed"])
        worst["worst"]["balanced_accuracy"] = 0.5
        self.assertFalse(gate_report(lock, canonical, bootstrap, surfaces, invariance, paired, worst)["passed"])


if __name__ == "__main__":
    unittest.main()
