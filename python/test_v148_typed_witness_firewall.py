import json
import unittest
from pathlib import Path

from v148_typed_witness_firewall import (
    build_valid_witnesses,
    evaluate,
    finalize_witness,
    malformed_witnesses,
    near_known_mutations,
)


class V148TypedWitnessFirewallTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = json.loads(Path("configs/v148-typed-witness-firewall.json").read_text())

    def test_llm_candidate_cannot_change_any_valid_witness_decision(self):
        for case in build_valid_witnesses(self.config):
            outputs = {
                finalize_witness(case["witness"], proposal, self.config)["final_state_id"]
                for proposal in self.config["outputIds"]
            }
            self.assertEqual(outputs, {case["truth_state_id"]})

    def test_malformed_witnesses_fail_closed(self):
        for witness in malformed_witnesses(self.config):
            output = finalize_witness(witness, "K31", self.config)
            self.assertFalse(output["witness_valid"])
            self.assertEqual(output["final_state_id"], "A00")
            self.assertTrue(output["final_output_structurally_valid"])

    def test_near_known_witnesses_cannot_be_forced_known_by_candidate(self):
        for row in near_known_mutations(self.config):
            output = finalize_witness(row["witness"], row["claimed_known_id"], self.config)
            self.assertNotIn(output["final_state_id"], self.config["knownIds"])

    def test_novel_candidate_never_defines_registers_or_executes_capability(self):
        novel = next(row for row in build_valid_witnesses(self.config) if row["truth_state_id"] == "N00")
        output = finalize_witness(novel["witness"], "K31", self.config)
        self.assertEqual(output["final_state_id"], "N00")
        self.assertFalse(output["capability_defined_or_registered"])
        self.assertFalse(output["executable"])
        self.assertEqual(output["actual_execution_count"], 0)

    def test_exhaustive_model_free_evaluation_passes(self):
        result = evaluate(self.config)
        self.assertTrue(result["passed"])
        self.assertTrue(all(result["checks"].values()))
        self.assertEqual(result["metrics"]["oracle_candidate_cross_product_count"], 98)


if __name__ == "__main__":
    unittest.main()
