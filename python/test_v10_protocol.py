import unittest

from v10_protocol import derive_allowed_values, folds


class V10ProtocolTests(unittest.TestCase):
    def test_allowed_values_are_conservative_outside_reliable_current_relations(self):
        self.assertEqual(derive_allowed_values("CURRENT", ["ENTAILED", "CONTRADICTED"]), ["active"])
        self.assertEqual(derive_allowed_values("CURRENT", ["CONTRADICTED", "ENTAILED"]), ["inactive"])
        self.assertEqual(derive_allowed_values("CURRENT", ["UNKNOWN", "UNKNOWN"]), ["inactive", "active"])
        self.assertEqual(derive_allowed_values("STALE_ONLY", ["ENTAILED", "CONTRADICTED"]), ["inactive", "active"])

    def test_twenty_four_folds_are_nonempty_and_disjoint(self):
        mechanics = ["hatch", "beacon", "pressure", "generator", "power", "rejection"]
        operators = {name: "binary_partition" if index < 3 else "multiway_partition" for index, name in enumerate(mechanics)}
        templates = ["a", "b", "c", "d", "e", "f"]
        lexicons = ["canonical", "entity_renamed", "paraphrased"]
        records = [
            {
                "split": split,
                "mechanic": mechanic,
                "operator_family": operators[mechanic],
                "template_family": template,
                "state_lexicon_family": lexicon,
            }
            for split in ("train", "evaluation")
            for mechanic in mechanics
            for template in templates
            for lexicon in lexicons
        ]
        values = folds(records)
        self.assertEqual(len(values), 24)
        self.assertEqual(
            {fold["kind"] for fold in values},
            {"context", "mechanic", "template", "lexicon", "operator", "combined"},
        )
        for fold in values:
            self.assertTrue(fold["train"].any())
            self.assertTrue(fold["evaluation"].any())
            self.assertFalse((fold["train"] & fold["evaluation"]).any())


if __name__ == "__main__":
    unittest.main()
