import unittest

from evaluate_v10_frozen import decision_from_gates, gate_report, relations_to_current, token_f1


def metrics(value=0.8):
    ablation = {
        "polarity_accuracy": value,
        "hypothesis_pair_consistency": value,
        "allowed_values_accuracy": value,
        "complete_flip_pair_accuracy": value,
        "symbolic_identifiability": {"balanced_accuracy": value},
    }
    return {
        "span_accuracy": value,
        "temporal_accuracy_predicted_span": value,
        "ablations": {
            "oracle_span_oracle_temporal": dict(ablation),
            "fully_predicted": dict(ablation),
        },
    }


class V10FrozenTests(unittest.TestCase):
    def test_relation_pairs_map_only_complementary_current_states(self):
        self.assertEqual(relations_to_current(["ENTAILED", "CONTRADICTED"]), "active")
        self.assertEqual(relations_to_current(["CONTRADICTED", "ENTAILED"]), "inactive")
        self.assertIsNone(relations_to_current(["UNKNOWN", "UNKNOWN"]))
        self.assertIsNone(relations_to_current(["ENTAILED", "ENTAILED"]))

    def test_token_f1_rewards_partial_and_exact_overlap(self):
        self.assertEqual(token_f1("the hatch is open", "the hatch is open"), 1.0)
        self.assertGreater(token_f1("hatch open", "the hatch is open"), 0.5)
        self.assertEqual(token_f1("generator", "mirror"), 0.0)

    def test_surface_and_complete_group_gates_are_hard(self):
        gates = {
            "minimumEveryFoldSpanAccuracy": 0.65,
            "minimumEverySurfaceSpanAccuracy": 0.6,
            "minimumEveryFoldTemporalAccuracy": 0.7,
            "minimumEverySurfaceTemporalAccuracy": 0.65,
            "minimumEveryFoldOraclePolarityAccuracy": 0.7,
            "minimumEverySurfaceOraclePolarityAccuracy": 0.65,
            "minimumEveryFoldNliPairConsistency": 0.7,
            "minimumEverySurfaceNliPairConsistency": 0.65,
            "minimumEveryFoldAllowedValuesAccuracy": 0.65,
            "minimumEverySurfaceAllowedValuesAccuracy": 0.6,
            "minimumEveryFoldSymbolicBalancedAccuracy": 0.65,
            "minimumEverySurfaceSymbolicBalancedAccuracy": 0.6,
            "minimumEveryFoldCompleteFlipPairAccuracy": 0.6,
            "minimumEveryFoldCompleteInterventionGroupAccuracy": 0.5,
        }
        results = {
            "context": {
                "overall": metrics(),
                "by_surface": {"canonical": metrics(0.59)},
                "group_scope": {"complete_intervention_group_accuracy": 0.8},
            }
        }
        report = gate_report(results, gates)
        self.assertFalse(report["passed"])
        self.assertIn("minimum_surface_span_accuracy", [value["name"] for value in report["checks"] if not value["passed"]])

    def test_decision_routes_oracle_polarity_failure_to_frozen_scale_only(self):
        report = {"passed": False, "checks": [
            {"name": "minimum_fold_oracle_polarity_accuracy", "passed": False}
        ]}
        self.assertEqual(
            decision_from_gates(report),
            "authorize_separately_locked_larger_frozen_capacity_diagnostic",
        )


if __name__ == "__main__":
    unittest.main()
