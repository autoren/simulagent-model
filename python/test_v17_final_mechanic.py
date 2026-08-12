import copy
import unittest

from evaluate_v10_frozen import gate_report


GATES = {
    "minimumEveryFoldSpanAccuracy": 0.65,
    "minimumEverySurfaceSpanAccuracy": 0.60,
    "minimumEveryFoldTemporalAccuracy": 0.70,
    "minimumEverySurfaceTemporalAccuracy": 0.65,
    "minimumEveryFoldOraclePolarityAccuracy": 0.70,
    "minimumEverySurfaceOraclePolarityAccuracy": 0.65,
    "minimumEveryFoldNliPairConsistency": 0.70,
    "minimumEverySurfaceNliPairConsistency": 0.65,
    "minimumEveryFoldAllowedValuesAccuracy": 0.65,
    "minimumEverySurfaceAllowedValuesAccuracy": 0.60,
    "minimumEveryFoldSymbolicBalancedAccuracy": 0.65,
    "minimumEverySurfaceSymbolicBalancedAccuracy": 0.60,
    "minimumEveryFoldCompleteFlipPairAccuracy": 0.60,
    "minimumEveryFoldCompleteInterventionGroupAccuracy": 0.50,
}


def cell(value: float = 0.9) -> dict:
    return {
        "span_accuracy": value,
        "temporal_accuracy_predicted_span": value,
        "ablations": {
            "oracle_span_oracle_temporal": {
                "polarity_accuracy": value,
                "hypothesis_pair_consistency": value,
            },
            "fully_predicted": {
                "allowed_values_accuracy": value,
                "symbolic_identifiability": {"balanced_accuracy": value},
                "complete_flip_pair_accuracy": value,
            },
        },
    }


def template_results() -> dict:
    return {
        f"template_{index}": {
            "overall": cell(),
            "by_surface": {name: cell() for name in ("canonical", "entity_renamed", "paraphrased")},
            "group_scope": {"complete_intervention_group_accuracy": 0.9},
        }
        for index in range(9)
    }


class V17FinalGateTests(unittest.TestCase):
    def test_exact_nine_by_three_topology_passes_unchanged_v15_gates(self) -> None:
        result = gate_report(template_results(), GATES)
        self.assertTrue(result["passed"])
        self.assertEqual(len(result["checks"]), 14)

    def test_one_lexicon_cell_failure_cannot_be_hidden_by_means(self) -> None:
        values = copy.deepcopy(template_results())
        values["template_3"]["by_surface"]["paraphrased"]["ablations"]["fully_predicted"][
            "symbolic_identifiability"
        ]["balanced_accuracy"] = 0.59
        result = gate_report(values, GATES)
        self.assertFalse(result["passed"])
        failed = [value for value in result["checks"] if not value["passed"]]
        self.assertEqual([value["name"] for value in failed], ["minimum_surface_symbolic_balanced_accuracy"])


if __name__ == "__main__":
    unittest.main()
