import json
import unittest
from pathlib import Path

from v148_typed_witness_firewall import finalize_witness
from v155_fresh_deterministic_question_retrieval_population import (
    ANSWER_STAGES,
    REQUEST_STAGES,
    audit_population,
    build_population,
    malformed_answer_events,
    witness_from_answer_event,
)


class V155FreshDeterministicQuestionRetrievalPopulationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = json.loads(
            Path("configs/v155-fresh-deterministic-question-retrieval-population.json").read_text()
        )
        cls.population = build_population(cls.config)

    def test_population_is_complete_and_balanced(self):
        summary = self.population["population_summary"]
        self.assertEqual(summary["fixture_count"], 288)
        self.assertEqual(summary["group_count"], 48)
        self.assertEqual(summary["request_fixture_count"], 192)
        self.assertEqual(summary["closed_answer_fixture_count"], 96)
        self.assertEqual(summary["split_counts"], {"development": 144, "evaluation": 144})
        self.assertTrue(all(count == 48 for count in summary["stage_counts"].values()))

    def test_catalog_exposes_only_explicit_retrieval_metadata(self):
        catalog = self.population["interaction_catalog"]
        self.assertEqual(len(catalog["queries"]), 6)
        for query in catalog["queries"]:
            self.assertEqual(
                set(query),
                {"query_id", "title", "question", "retrieval_profile", "options"},
            )
            self.assertEqual(
                set(query["retrieval_profile"]),
                {"anchor_phrases", "primary_terms", "secondary_terms"},
            )
            self.assertEqual(len(query["options"]), 2)

    def test_public_rows_hide_truth_and_candidate_surfaces(self):
        forbidden = {
            "group_id", "family_id", "stage", "truth_state_id", "compatible_state_ids",
            "oracle_query_id", "oracle_witness", "candidate_state_ids", "state_ranking",
            "llm_proposal", "confidence",
        }
        self.assertTrue(
            all(not (forbidden & set(row)) for row in self.population["public_fixtures"])
        )

    def test_typed_answers_route_and_preanswer_remains_fail_closed(self):
        catalog = self.population["interaction_catalog"]
        for row in self.population["hidden_fixtures"]:
            witness = witness_from_answer_event(row["closed_answer_event"], catalog)
            output = finalize_witness(witness, None, self.config)
            if row["stage"] in ANSWER_STAGES:
                self.assertTrue(output["witness_valid"])
                self.assertEqual(output["final_state_id"], row["truth_state_id"])
            elif row["stage"] in REQUEST_STAGES:
                self.assertFalse(output["witness_valid"])
                self.assertEqual(output["final_state_id"], "A00")
            self.assertFalse(output["authoritative_hypothesis_universe_pruned"])
            self.assertEqual(output["actual_execution_count"], 0)

    def test_malformed_answer_events_fail_closed(self):
        catalog = self.population["interaction_catalog"]
        for event in malformed_answer_events(catalog):
            output = finalize_witness(witness_from_answer_event(event, catalog), None, self.config)
            self.assertFalse(output["witness_valid"])
            self.assertEqual(output["final_state_id"], "A00")

    def test_full_audit_passes_without_prior_rows(self):
        result = audit_population(self.population, self.config, [])
        self.assertTrue(result["passed"], result)
        self.assertTrue(all(result["checks"].values()))
        self.assertEqual(result["candidate_proposal_field_count"], 0)
        self.assertEqual(result["model_generation_or_score_count"], 0)
        self.assertEqual(result["actual_execution_count"], 0)


if __name__ == "__main__":
    unittest.main()
