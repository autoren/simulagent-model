import json
import unittest
from pathlib import Path

from v140_thinking_qualification_gap import audit_gap


class V140ThinkingQualificationGapTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = json.loads(Path("configs/v140-thinking-qualification-gap.json").read_text())
        cls.result = json.loads(Path("outputs/v139-repaired-direct-vs-thinking-realization/model-realization/result.json").read_text())
        cls.hidden = [
            row for row in json.loads(Path("outputs/v135-controlled-open-world-minimal-pairs/design/hidden-fixtures.json").read_text())
            if row["split"] == "development"
        ]
        cls.audit = audit_gap(cls.result, cls.hidden, cls.config)

    def test_exact_failed_gates_and_integer_gaps(self):
        self.assertEqual(self.audit["failed_thinking_gate_families"], ["ambiguous_abstention_accuracy", "structured_validity"])
        self.assertEqual(self.audit["structured_validity"]["minimum_additional_valid_outputs"], 2)
        self.assertEqual(self.audit["ambiguity"]["minimum_additional_apparent_correct"], 1)

    def test_invalids_are_ceiling_limited_and_semantic_gap_remains(self):
        self.assertEqual(self.audit["invalid_output_count"], 3)
        self.assertTrue(self.audit["all_invalid_at_condition_token_ceiling"])
        self.assertEqual(self.audit["ambiguity"]["valid_semantic_overcommitment_count"], 2)
        self.assertAlmostEqual(self.audit["ambiguity"]["valid_only_accuracy"], 16 / 18)

    def test_neither_single_mechanism_counterfactual_qualifies(self):
        self.assertFalse(self.audit["counterfactuals"]["completion_only"]["qualifies_both_failed_families"])
        self.assertFalse(self.audit["counterfactuals"]["semantic_only"]["qualifies_both_failed_families"])
        self.assertTrue(self.audit["minimum_joint_gap"]["both_mechanism_families_required"])

    def test_paired_thinking_gain_is_positive(self):
        paired = self.audit["paired_correctness"]
        self.assertEqual(paired["thinking_repairs_direct"], 4)
        self.assertEqual(paired["thinking_regresses_direct"], 1)
        self.assertEqual(paired["net_thinking_repairs"], 3)


if __name__ == "__main__":
    unittest.main()
