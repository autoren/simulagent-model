import json
import unittest
from pathlib import Path

from v135_controlled_open_world_minimal_pairs import build_catalog
from v136_controlled_clarification_value import best_action, decision_cost, evaluate, evaluate_gates


class V136ClarificationValueTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = json.loads(Path("configs/v136-controlled-clarification-value.json").read_text())
        cls.v135 = json.loads(Path("configs/v135-controlled-open-world-minimal-pairs.json").read_text())
        cls.catalog = build_catalog(cls.v135)
        cls.result = evaluate(cls.config, cls.v135, cls.catalog)

    def test_historical_asymmetric_cost_mapping(self):
        self.assertEqual(decision_cost("K01", "K01", self.catalog, self.config), 0.0)
        self.assertEqual(decision_cost("K01", "K02", self.catalog, self.config), 10.0)
        self.assertEqual(decision_cost("N01", "K01", self.catalog, self.config), 10.0)
        self.assertEqual(decision_cost("U00", "A00", self.catalog, self.config), 1.0)

    def test_all_preregistered_gates_pass(self):
        gates = evaluate_gates(self.result, self.config)
        self.assertTrue(all(gates.values()), gates)

    def test_targeted_question_reduces_required_reliability(self):
        self.assertAlmostEqual(self.result["worst_query_benefit_reliability_threshold"], 0.913)
        self.assertLess(self.result["worst_query_benefit_reliability_threshold"], 0.9725)

    def test_clear_truth_prefers_direct_exact_decision(self):
        for row in self.result["clear_rows"]:
            self.assertEqual(row["no_query_action"], row["truth_choice_id"])
            self.assertEqual(row["no_query_expected_cost"], 0.0)
            self.assertTrue(row["skip_query_preferred"])


if __name__ == "__main__":
    unittest.main()
