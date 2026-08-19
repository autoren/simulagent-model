import json
import unittest
from pathlib import Path

from v159_fresh_controlled_relational_grammar_population import build_population
from v160_model_free_controlled_relational_grammar_policy import (
    build_episodes,
    choose_initial_query,
    normalize_alias,
)


class V160ModelFreeControlledRelationalGrammarPolicyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = json.loads(
            Path("configs/v160-model-free-controlled-relational-grammar-policy.json").read_text()
        )
        cls.catalog = {
            "queries": [
                {
                    "query_id": "Q81",
                    "title": "urban ecology",
                    "question": "Which ecology work?",
                    "grammar_aliases": ["canopy works file", "roof habitat count"],
                    "retrieval_profile": {
                        "anchor_phrases": ["street tree"],
                        "primary_terms": ["street", "tree"],
                        "secondary_terms": ["works"],
                    },
                    "options": [
                        {"option_id": "LEFT", "text": "street tree"},
                        {"option_id": "RIGHT", "text": "roof count"},
                    ],
                },
                {
                    "query_id": "Q82",
                    "title": "harbor reservation",
                    "question": "Which harbor resource?",
                    "grammar_aliases": ["quay access credential", "shore supply slot"],
                    "retrieval_profile": {
                        "anchor_phrases": ["harbor berth"],
                        "primary_terms": ["harbor", "berth"],
                        "secondary_terms": ["reserve"],
                    },
                    "options": [
                        {"option_id": "LEFT", "text": "berth access"},
                        {"option_id": "RIGHT", "text": "power slot"},
                    ],
                },
            ]
        }

    def test_alias_normalization_is_unicode_case_and_space_stable(self):
        self.assertEqual(normalize_alias("  CANOPY   Works File  "), "canopy works file")
        self.assertEqual(normalize_alias("ＣＡＮＯＰＹ works file"), "canopy works file")

    def test_unique_registered_alias_selects_specific_query(self):
        fixture = {"conversation": [{"role": "user", "text": 'Case X relates only to "roof habitat count".'}]}
        result = choose_initial_query(fixture, self.catalog, self.config)
        self.assertEqual(result["initial_query_id"], "Q81")
        self.assertEqual(result["grammar_status"], "UNIQUE_REGISTERED_QUERY")
        self.assertEqual(result["decision_source"], "REGISTERED_RELATION_GRAMMAR")

    def test_cross_query_aliases_force_generic_route(self):
        fixture = {
            "conversation": [
                {
                    "role": "user",
                    "text": 'Case X relates to either "canopy works file" or "quay access credential".',
                }
            ]
        }
        result = choose_initial_query(fixture, self.catalog, self.config)
        self.assertEqual(result["initial_query_id"], "Q80")
        self.assertEqual(result["grammar_status"], "CROSS_QUERY_CONFLICT")
        self.assertEqual(result["grammar_query_count"], 2)

    def test_unknown_alias_forces_generic_route(self):
        fixture = {"conversation": [{"role": "user", "text": 'The sole relation is "unregistered relation".'}]}
        result = choose_initial_query(fixture, self.catalog, self.config)
        self.assertEqual(result["initial_query_id"], "Q80")
        self.assertEqual(result["grammar_status"], "UNKNOWN_ALIAS")
        self.assertEqual(result["unknown_alias_count"], 1)

    def test_no_quoted_surface_uses_strict_retrieval(self):
        fixture = {"conversation": [{"role": "user", "text": "Authorize street tree works"}]}
        result = choose_initial_query(fixture, self.catalog, self.config)
        self.assertEqual(result["initial_query_id"], "Q81")
        self.assertEqual(result["decision_source"], "STRICT_RETRIEVAL")

    def test_episode_builder_has_one_episode_per_request(self):
        population_config = json.loads(
            Path("configs/v159-fresh-controlled-relational-grammar-population.json").read_text()
        )
        population = build_population(population_config)
        metadata = [row for row in population["hidden_fixtures"] if row["split"] == "development"]
        episodes = build_episodes(metadata)
        self.assertEqual(len(episodes), 64)
        self.assertEqual(sum(row["side"] == "left" for row in episodes), 16)
        self.assertEqual(sum(row["side"] == "right" for row in episodes), 16)
        self.assertEqual(sum(row["side"] == "unclear" for row in episodes), 32)


if __name__ == "__main__":
    unittest.main()
