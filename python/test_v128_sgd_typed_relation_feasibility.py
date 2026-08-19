import json
import unittest
from pathlib import Path

from v128_sgd_typed_relation_feasibility import (
    build_support_signatures, population_gates, relation_tokens, select_fresh_population,
)


class V128TypedRelationTests(unittest.TestCase):
    def test_fresh_population_excludes_both_prior_sets(self):
        config = json.loads(Path("configs/v128-sgd-typed-relation-feasibility.json").read_text())
        inventory = json.loads(Path(config["sourceInventory"]).read_text())
        catalog = json.loads(Path(config["choiceCatalog"]).read_text())
        excluded = [json.loads(Path(path).read_text()) for path in config["excludedEvaluationPopulations"]]
        population = select_fresh_population(inventory, excluded, catalog, config)
        self.assertTrue(all(population_gates(population, config).values()))
        self.assertNotIn("utterance", set().union(*(row.keys() for row in population["records"])))

    def test_relation_tokens_do_not_read_or_encode_values(self):
        frame = {
            "state": {"slot_values": {"date": ["secret"]}, "requested_slots": ["price"]},
            "actions": [
                {"act": "INFORM_INTENT", "slot": "intent", "values": ["hidden"]},
                {"act": "INFORM", "slot": "date", "values": ["secret"]},
            ],
        }
        tokens = relation_tokens(frame)
        self.assertEqual(tokens, {"STATE_SLOT::date", "REQUEST_SLOT::price", "ACTION::INFORM::date"})
        self.assertFalse(any("secret" in token or "hidden" in token for token in tokens))

    def test_support_is_union_without_frequencies(self):
        signatures = build_support_signatures([("a", {"x"}), ("a", {"x", "y"}), ("b", {"z"})])
        self.assertEqual(signatures["a"]["allowed"], {"x", "y"})
        self.assertEqual(signatures["a"]["required"], set())


if __name__ == "__main__":
    unittest.main()
