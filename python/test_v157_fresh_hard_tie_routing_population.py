import json
import unittest
from pathlib import Path

from v148_typed_witness_firewall import finalize_witness
from v157_fresh_hard_tie_routing_population import (
    REQUEST_STAGES,
    ROUTE_STAGES,
    SPECIFIC_ANSWER_STAGES,
    audit_population,
    build_population,
    malformed_answer_events,
    route_from_answer_event,
    witness_from_answer_event,
)


class V157FreshHardTieRoutingPopulationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = json.loads(Path("configs/v157-fresh-hard-tie-routing-population.json").read_text())
        cls.population = build_population(cls.config)
        cls.catalog = cls.population["interaction_catalog"]

    def test_population_counts_and_strata_are_complete(self):
        summary = self.population["population_summary"]
        self.assertEqual(summary["fixture_count"], 384)
        self.assertEqual(summary["group_count"], 48)
        self.assertEqual(summary["split_counts"], {"development": 192, "evaluation": 192})
        self.assertTrue(all(count == 48 for count in summary["stage_counts"].values()))
        self.assertTrue(all(count == 48 for count in summary["stratum_counts"].values()))

    def test_generic_query_routes_but_cannot_create_semantic_witness(self):
        generic = next(row for row in self.catalog["queries"] if row["query_id"] == "Q70")
        self.assertEqual(generic["query_kind"], "GENERIC_ROUTE")
        self.assertEqual(len(generic["options"]), 7)
        for option in generic["options"]:
            self.assertNotIn("state_id", option)
            self.assertNotIn("witness", option)
            event = {"query_id": "Q70", "selected_option_id": option["option_id"]}
            self.assertTrue(route_from_answer_event(event, self.catalog)["route_valid"])
            self.assertIsNone(witness_from_answer_event(event, self.catalog))
            output = finalize_witness(None, None, self.config)
            self.assertEqual(output["final_state_id"], "A00")

    def test_specific_answers_create_exact_trusted_witnesses(self):
        for row in self.population["hidden_fixtures"]:
            if row["stage"] not in SPECIFIC_ANSWER_STAGES:
                continue
            witness = witness_from_answer_event(row["closed_answer_event"], self.catalog)
            output = finalize_witness(witness, None, self.config)
            self.assertTrue(output["witness_valid"])
            self.assertEqual(output["final_state_id"], row["truth_state_id"])
            self.assertEqual(output["actual_execution_count"], 0)

    def test_requests_and_routes_remain_semantically_fail_closed(self):
        for row in self.population["hidden_fixtures"]:
            if row["stage"] not in REQUEST_STAGES | ROUTE_STAGES:
                continue
            witness = witness_from_answer_event(row["closed_answer_event"], self.catalog)
            output = finalize_witness(witness, None, self.config)
            self.assertFalse(output["witness_valid"])
            self.assertEqual(output["final_state_id"], "A00")
            self.assertFalse(output["authoritative_hypothesis_universe_pruned"])

    def test_malformed_events_neither_route_nor_witness(self):
        for event in malformed_answer_events(self.catalog):
            self.assertFalse(route_from_answer_event(event, self.catalog)["route_valid"])
            self.assertIsNone(witness_from_answer_event(event, self.catalog))

    def test_public_rows_hide_strata_truth_routes_and_oracles(self):
        forbidden = {
            "group_id", "family_id", "stage", "stratum", "truth_state_id",
            "compatible_state_ids", "oracle_specific_query_id", "oracle_initial_query_id",
            "route_target_query_id", "oracle_route", "oracle_witness", "candidate_state_ids",
            "state_ranking", "llm_proposal", "confidence",
        }
        self.assertTrue(all(not (forbidden & set(row)) for row in self.population["public_fixtures"]))

    def test_full_audit_passes_without_prior_rows(self):
        result = audit_population(self.population, self.config, [])
        self.assertTrue(result["passed"], result)
        self.assertTrue(all(result["checks"].values()))
        self.assertEqual(result["generic_route_semantic_witness_count"], 0)
        self.assertEqual(result["policy_score_count"], 0)


if __name__ == "__main__":
    unittest.main()
