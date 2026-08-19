import json
import unittest
from pathlib import Path

from v141_two_stage_controller_feasibility import (
    evaluate,
    evaluate_gates,
    evaluate_point,
    frechet_all_lower,
)


class V141TwoStageControllerFeasibilityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = json.loads(Path("configs/v141-two-stage-controller-feasibility.json").read_text())
        cls.v135 = json.loads(Path("configs/v135-controlled-open-world-minimal-pairs.json").read_text())
        cls.v136 = json.loads(Path("configs/v136-controlled-clarification-value.json").read_text())
        cls.catalog = json.loads(Path("outputs/v135-controlled-open-world-minimal-pairs/design/choice-catalog.json").read_text())
        cls.result = evaluate(cls.config, cls.v135, cls.catalog, cls.v136)

    def test_frechet_bound_does_not_multiply_stage_reliabilities(self):
        self.assertAlmostEqual(frechet_all_lower([0.99, 0.99, 0.99]), 0.97)
        self.assertNotAlmostEqual(frechet_all_lower([0.99, 0.99, 0.99]), 0.99**3)

    def test_reference_point_passes_conservative_bounded_gates(self):
        reference = self.result["reference"]
        self.assertTrue(reference["qualified_on_bounded_gates"], reference)
        self.assertAlmostEqual(reference["metrics"]["structured_validity"], 0.99)
        self.assertAlmostEqual(reference["metrics"]["ambiguous_abstention_lower_bound"], 0.99)
        self.assertAlmostEqual(reference["metrics"]["clear_exact_accuracy_lower_bound"], 0.97)
        self.assertAlmostEqual(reference["metrics"]["full_five_stage_group_accuracy_lower_bound"], 0.87)
        self.assertLessEqual(reference["metrics"]["sequential_mean_decision_cost_upper_bound"], 0.6470000001)
        self.assertLessEqual(reference["metrics"]["sequential_false_known_on_right_truth_upper_bound"], 0.0400000001)

    def test_thresholds_are_exact_on_frozen_grid(self):
        self.assertEqual(self.result["symmetric_marginal_reliability_threshold"], 0.99)
        self.assertEqual(
            self.result["individual_thresholds_with_other_marginals_at_reference"],
            {
                "finalizer_validity": 0.99,
                "ambiguity_sensitivity": 0.98,
                "decidable_specificity": 0.985,
                "proposal_correctness": 0.985,
            },
        )

    def test_subthreshold_symmetric_point_fails(self):
        point = evaluate_point(0.989, 0.989, 0.989, 0.989, self.config, self.v135, self.catalog, self.v136)
        self.assertFalse(point["qualified_on_bounded_gates"])
        self.assertFalse(point["gates"]["structured_validity"])

    def test_no_independence_or_candidate_attraction_claim(self):
        gates = evaluate_gates(self.result, self.config)
        self.assertTrue(all(gates.values()), gates)
        self.assertFalse(self.result["independence_assumption_used"])
        self.assertFalse(self.result["candidate_attraction_certified"])


if __name__ == "__main__":
    unittest.main()
