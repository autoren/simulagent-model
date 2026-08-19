import json
import unittest
from pathlib import Path

from v148_typed_witness_firewall import finalize_witness
from v152_fresh_question_order_population import (
    ANSWER_STAGES,
    CANDIDATE_PROPOSAL_FIELDS,
    STAGES,
    audit_population,
    build_population,
    malformed_answer_events,
    witness_from_answer_event,
)


class V152FreshQuestionOrderPopulationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = json.loads(Path("configs/v152-fresh-question-order-population.json").read_text())
        cls.population = build_population(cls.config)
        prior_paths = [
            Path("outputs/v135-controlled-open-world-minimal-pairs/design/public-fixtures.json"),
            Path("outputs/v142-certificate-interface-population/design/public-fixtures.json"),
            Path("outputs/v146-fresh-codebook-population/design/public-fixtures.json"),
            Path("outputs/v149-fresh-closed-interaction-population/design/public-fixtures.json"),
        ]
        cls.prior = [row for path in prior_paths for row in json.loads(path.read_text())]

    def test_population_counts_splits_and_stages(self):
        summary = self.population["population_summary"]
        self.assertEqual(summary["fixture_count"], 288)
        self.assertEqual(summary["group_count"], 48)
        self.assertEqual(summary["request_fixture_count"], 192)
        self.assertEqual(summary["closed_answer_fixture_count"], 96)
        self.assertEqual(summary["split_counts"], {"development": 144, "evaluation": 144})
        self.assertEqual(set(summary["stage_counts"]), set(STAGES))

    def test_public_rows_hide_truth_and_candidate_proposals(self):
        forbidden = {
            "group_id", "family_id", "stage", "language_class", "truth_state_id",
            "compatible_state_ids", "oracle_query_id", "oracle_witness",
            "trusted_witness_available", "variant_index",
        }
        public = self.population["public_fixtures"]
        hidden = {row["fixture_id"]: row for row in self.population["hidden_fixtures"]}
        self.assertTrue(all(not (forbidden & set(row)) for row in public))
        self.assertTrue(all(not (CANDIDATE_PROPOSAL_FIELDS & set(row)) for row in public))
        self.assertTrue(all(not (CANDIDATE_PROPOSAL_FIELDS & set(row)) for row in hidden.values()))
        self.assertTrue(
            all(
                (row["closed_answer_event"] is not None)
                == (hidden[row["fixture_id"]]["stage"] in ANSWER_STAGES)
                for row in public
            )
        )

    def test_registered_answers_route_without_any_model_state_proposal(self):
        catalog = self.population["interaction_catalog"]
        for row in self.population["hidden_fixtures"]:
            if row["stage"] not in ANSWER_STAGES:
                continue
            witness = witness_from_answer_event(row["closed_answer_event"], catalog)
            output = finalize_witness(witness, None, self.config)
            self.assertEqual(output["final_state_id"], row["truth_state_id"])
            self.assertTrue(output["llm_proposal_non_authoritative"])
            self.assertFalse(output["capability_defined_or_registered"])
            self.assertFalse(output["executable"])

    def test_malformed_answer_events_fail_closed(self):
        catalog = self.population["interaction_catalog"]
        for event in malformed_answer_events(catalog):
            output = finalize_witness(witness_from_answer_event(event, catalog), None, self.config)
            self.assertFalse(output["witness_valid"])
            self.assertEqual(output["final_state_id"], "A00")

    def test_no_exact_prior_conversation_overlap(self):
        result = audit_population(self.population, self.config, self.prior)
        self.assertEqual(result["exact_prior_conversation_overlap_count"], 0)

    def test_all_population_gates_pass(self):
        result = audit_population(self.population, self.config, self.prior)
        self.assertTrue(result["passed"], result)


if __name__ == "__main__":
    unittest.main()
