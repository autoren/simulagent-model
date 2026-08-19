import json
import unittest
from pathlib import Path

from v143_oracle_certificate_policy import evaluate, evaluate_gates, malformed_mutations, oracle_certificate


class V143OracleCertificatePolicyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = json.loads(Path("configs/v143-oracle-certificate-policy.json").read_text())
        cls.v142 = json.loads(Path("configs/v142-certificate-interface-population.json").read_text())
        cls.hidden = json.loads(Path("outputs/v142-certificate-interface-population/design/hidden-fixtures.json").read_text())
        cls.catalog = json.loads(Path("outputs/v142-certificate-interface-population/design/choice-catalog.json").read_text())
        cls.v136 = json.loads(Path("configs/v136-controlled-clarification-value.json").read_text())
        cls.result = evaluate(cls.config, cls.hidden, cls.catalog, cls.v142, cls.v136)

    def test_oracle_certificate_matches_stage_semantics(self):
        for row in self.hidden:
            certificate = oracle_certificate(row)
            self.assertEqual(certificate["compatible_choice_ids"], row["compatible_choice_ids"])
            if row["truth_choice_id"] == "A00":
                self.assertEqual(certificate["evidence_status"], "INSUFFICIENT")
                self.assertEqual(certificate["proposed_choice_id"], "A00")
            else:
                self.assertEqual(certificate["evidence_status"], "SUFFICIENT")
                self.assertEqual(certificate["proposed_choice_id"], row["truth_choice_id"])

    def test_oracle_and_sequential_metrics_are_exact(self):
        metrics = self.result["metrics"]
        self.assertEqual(metrics["final_choice_exact_accuracy"], 1.0)
        self.assertEqual(metrics["ambiguous_query_rate"], 1.0)
        self.assertAlmostEqual(metrics["sequential_mean_decision_cost"], 0.3)
        self.assertAlmostEqual(metrics["worst_family_sequential_improvement"], 0.7)
        self.assertEqual(metrics["sequential_false_known_on_right_truth"], 0.0)

    def test_malformed_mutations_all_fail_closed(self):
        self.assertEqual(len(malformed_mutations()), 9)
        self.assertEqual(self.result["metrics"]["malformed_mutation_fail_closed_rate"], 1.0)

    def test_valid_wrong_singleton_is_not_semantically_detectable(self):
        limitation = self.result["valid_wrong_singleton_limitation"]
        self.assertTrue(limitation["structurally_valid"])
        self.assertTrue(limitation["semantic_truth_not_checkable_by_interface"])

    def test_all_noncompensatory_gates_pass(self):
        gates = evaluate_gates(self.result, self.config)
        self.assertTrue(all(gates.values()), gates)


if __name__ == "__main__":
    unittest.main()
