from __future__ import annotations

import unittest

from v116_typed_clarification_voi import answer_distribution, prior_distribution, truth_choice


CATALOG = {"choices": [
    *[
        {"choice_id": f"K{i:02d}", "kind": "KNOWN", "intent_id": f"s::i{i}", "scenario": "s"}
        for i in range(12)
    ],
    *[
        {"choice_id": f"N{i:02d}", "kind": "NOVEL", "scenario": scenario}
        for i, scenario in enumerate(("s", "t", "u"))
    ],
    {"choice_id": "U00", "kind": "UNSUPPORTED"},
    {"choice_id": "A00", "kind": "ABSTAIN"},
]}


class V116TypedClarificationTests(unittest.TestCase):
    def test_answer_channel_is_normalized(self) -> None:
        identifiers = [row["choice_id"] for row in CATALOG["choices"]]
        for truth in identifiers:
            for reliability in (0.7, 0.95, 1.0):
                distribution = answer_distribution(truth, reliability, identifiers, "A00", 0.5)
                self.assertAlmostEqual(sum(distribution.values()), 1.0)
                self.assertEqual(distribution[truth], reliability)

    def test_prior_is_normalized(self) -> None:
        identifiers = [row["choice_id"] for row in CATALOG["choices"]]
        for probability in (1 / 17, 0.5, 0.75):
            distribution = prior_distribution("K00", probability, identifiers)
            self.assertAlmostEqual(sum(distribution.values()), 1.0)
            self.assertEqual(distribution["K00"], probability)

    def test_truth_mapping_uses_structural_metadata(self) -> None:
        self.assertEqual(truth_choice({"class_label": "known_familiar", "scenario": "s", "intent": "i1"}, CATALOG), "K01")
        self.assertEqual(truth_choice({"class_label": "novel_valid", "scenario": "t", "intent": "x"}, CATALOG), "N01")
        self.assertEqual(truth_choice({"class_label": "unsupported", "scenario": "z", "intent": "x"}, CATALOG), "U00")


if __name__ == "__main__":
    unittest.main()
