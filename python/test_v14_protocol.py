import json
import unittest
from pathlib import Path

from v14_protocol import primary_folds, zero_shot_operator_folds


class V14ProtocolTests(unittest.TestCase):
    def test_locked_fold_taxonomy(self):
        records = []
        for split in ("train", "evaluation"):
            for mechanic, operator in (
                ("a", "binary_partition"), ("b", "binary_partition"), ("c", "binary_partition"),
                ("d", "multiway_partition"), ("e", "multiway_partition"), ("f", "multiway_partition"),
            ):
                for semantic in ("affirmative_gold", "negated_opposite", "contrastive_both"):
                    for surface in (f"{semantic}_1", f"{semantic}_2", f"{semantic}_3"):
                        for lexicon in ("canonical", "entity_renamed", "paraphrased"):
                            records.append({
                                "split": split, "mechanic": mechanic, "operator_family": operator,
                                "semantic_operator_family": semantic, "template_family": surface,
                                "state_lexicon_family": lexicon,
                            })
        primary = primary_folds(records)
        diagnostics = zero_shot_operator_folds(records)
        self.assertEqual(len(primary), 27)
        self.assertEqual(len([fold for fold in primary if fold["kind"] == "surface"]), 9)
        self.assertEqual(len(diagnostics), 3)
        self.assertTrue(all(fold["kind"] == "semantic_operator_diagnostic" for fold in diagnostics))


if __name__ == "__main__":
    unittest.main()
