import json
import unittest
from pathlib import Path

from v148_typed_witness_firewall import finalize_witness
from v159_fresh_controlled_relational_grammar_population import (
    STAGES,
    audit_population,
    build_population,
    route_from_answer_event,
    witness_from_answer_event,
)


class V159FreshControlledRelationalGrammarPopulationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = json.loads(
            Path("configs/v159-fresh-controlled-relational-grammar-population.json").read_text()
        )
        cls.population = build_population(cls.config)

    def test_population_is_balanced_and_group_complete(self):
        summary = self.population["population_summary"]
        self.assertEqual(summary["group_count"], 32)
        self.assertEqual(summary["fixture_count"], 256)
        self.assertEqual(summary["split_counts"], {"development": 128, "evaluation": 128})
        self.assertTrue(all(count == 32 for count in summary["stage_counts"].values()))
        self.assertEqual(set(summary["stage_counts"]), set(STAGES))

    def test_public_catalog_aliases_are_unique_and_state_free(self):
        queries = [
            row
            for row in self.population["interaction_catalog"]["queries"]
            if row["query_kind"] == "SPECIFIC_WITNESS"
        ]
        aliases = [alias for row in queries for alias in row["grammar_aliases"]]
        self.assertEqual(len(aliases), 8)
        self.assertEqual(len(set(aliases)), 8)
        self.assertTrue(
            all(not ({"state_id", "choice_id", "witness", "truth_state_id"} & set(row)) for row in queries)
        )

    def test_unique_and_conflict_relations_have_distinct_hidden_cardinality(self):
        rows = self.population["hidden_fixtures"]
        unique = [row for row in rows if row["stage"] == "request_grammar_unique"]
        conflict = [row for row in rows if row["stage"] == "request_grammar_conflict"]
        self.assertTrue(all(len(row["grammar_relation_aliases"]) == 1 for row in unique))
        self.assertTrue(all(len(row["grammar_query_ids"]) == 1 for row in unique))
        self.assertTrue(all(len(row["grammar_relation_aliases"]) == 2 for row in conflict))
        self.assertTrue(all(len(row["grammar_query_ids"]) == 2 for row in conflict))
        self.assertTrue(all(row["oracle_initial_query_id"] == "Q80" for row in conflict))

    def test_generic_route_never_supplies_semantic_witness(self):
        catalog = self.population["interaction_catalog"]
        route_rows = [
            row for row in self.population["hidden_fixtures"] if row["stage"].startswith("closed_route_")
        ]
        self.assertTrue(
            all(route_from_answer_event(row["closed_answer_event"], catalog)["route_valid"] for row in route_rows)
        )
        self.assertTrue(
            all(witness_from_answer_event(row["closed_answer_event"], catalog) is None for row in route_rows)
        )

    def test_specific_answer_is_the_only_semantic_witness_path(self):
        catalog = self.population["interaction_catalog"]
        rows = [
            row for row in self.population["hidden_fixtures"] if row["stage"].startswith("closed_specific_")
        ]
        outputs = [
            finalize_witness(
                witness_from_answer_event(row["closed_answer_event"], catalog), None, self.config
            )
            for row in rows
        ]
        self.assertTrue(
            all(output["final_state_id"] == rows[index]["truth_state_id"] for index, output in enumerate(outputs))
        )
        self.assertTrue(all(output["actual_execution_count"] == 0 for output in outputs))

    def test_full_population_audit_passes(self):
        result = audit_population(self.population, self.config, [])
        self.assertTrue(result["passed"], result["checks"])
        self.assertEqual(result["grammar_alias_uniqueness"], 1.0)
        self.assertEqual(result["conflict_distinct_query_rate"], 1.0)
        self.assertEqual(result["candidate_proposal_field_count"], 0)


if __name__ == "__main__":
    unittest.main()
