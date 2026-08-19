import json
import unittest
from pathlib import Path

from v158_model_free_hard_tie_routing_policy import build_episodes, choose_initial_query


class V158ModelFreeHardTieRoutingPolicyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = json.loads(Path("configs/v158-model-free-hard-tie-routing-policy.json").read_text())
        cls.catalog = {
            "queries": [
                {
                    "query_id": "Q71", "title": "identify_collection_service",
                    "question": "Which collection service?",
                    "retrieval_profile": {"anchor_phrases": ["compost container"], "primary_terms": ["compost", "container"], "secondary_terms": ["replace"]},
                    "options": [{"option_id": "LEFT", "text": "A compost container"}, {"option_id": "RIGHT", "text": "A food pickup"}],
                },
                {
                    "query_id": "Q72", "title": "identify_transport_deposit",
                    "question": "Which transport deposit?",
                    "retrieval_profile": {"anchor_phrases": ["rail pass"], "primary_terms": ["rail", "pass"], "secondary_terms": ["refund"]},
                    "options": [{"option_id": "LEFT", "text": "A rail pass"}, {"option_id": "RIGHT", "text": "A bike deposit"}],
                },
            ]
        }

    def test_high_score_and_margin_select_specific(self):
        fixture = {"conversation": [{"role": "user", "text": "Replace compost container C-1"}]}
        result = choose_initial_query(fixture, self.catalog, self.config)
        self.assertEqual(result["initial_query_id"], "Q71")
        self.assertTrue(result["specific_selected"])

    def test_low_score_uses_generic_route(self):
        fixture = {"conversation": [{"role": "user", "text": "Change the surplus-meal vessel"}]}
        result = choose_initial_query(fixture, self.catalog, self.config)
        self.assertEqual(result["initial_query_id"], "Q70")
        self.assertFalse(result["specific_selected"])

    def test_equal_high_scores_use_generic_even_with_source_tie_break(self):
        fixture = {"conversation": [{"role": "user", "text": "Replace compost container and refund rail pass"}]}
        result = choose_initial_query(fixture, self.catalog, self.config)
        self.assertEqual(result["initial_query_id"], "Q70")
        self.assertGreater(result["top_score_tie_count"], 1)

    def test_episode_builder_has_left_right_and_unclear_paths(self):
        base = {"group_id": "g", "oracle_specific_query_id": "Q71", "route_target_query_id": "Q71"}
        rows = [
            {**base, "fixture_id": "a", "stage": "request_lexical_control", "stratum": "lexical_control", "truth_state_id": "K71", "oracle_initial_query_id": "Q71", "closed_answer_event": None},
            {**base, "fixture_id": "b", "stage": "request_uncatalogued_paraphrase", "stratum": "uncatalogued_paraphrase", "truth_state_id": "N70", "oracle_initial_query_id": "Q70", "closed_answer_event": None},
            {**base, "fixture_id": "c", "stage": "request_relational_tie", "stratum": "relational_tie", "truth_state_id": "A00", "oracle_initial_query_id": "Q70", "closed_answer_event": None},
            {**base, "fixture_id": "d", "stage": "request_insufficient", "stratum": "insufficient", "truth_state_id": "A00", "oracle_initial_query_id": "Q70", "closed_answer_event": None},
            {**base, "fixture_id": "rf", "stage": "closed_route_family", "stratum": "trusted_route_family", "truth_state_id": "A00", "oracle_initial_query_id": None, "closed_answer_event": {"query_id": "Q70", "selected_option_id": "ROUTE_Q71"}},
            {**base, "fixture_id": "ru", "stage": "closed_route_unclear", "stratum": "trusted_route_unclear", "truth_state_id": "A00", "oracle_initial_query_id": None, "closed_answer_event": {"query_id": "Q70", "selected_option_id": "UNCLEAR"}},
            {**base, "fixture_id": "l", "stage": "closed_specific_left", "stratum": "trusted_specific_left", "truth_state_id": "K71", "oracle_initial_query_id": None, "closed_answer_event": {"query_id": "Q71", "selected_option_id": "LEFT"}},
            {**base, "fixture_id": "r", "stage": "closed_specific_right", "stratum": "trusted_specific_right", "truth_state_id": "N70", "oracle_initial_query_id": None, "closed_answer_event": {"query_id": "Q71", "selected_option_id": "RIGHT"}},
        ]
        episodes = build_episodes(rows)
        self.assertEqual(len(episodes), 5)
        self.assertEqual(sum(row["side"] == "unclear" for row in episodes), 1)
        self.assertEqual(sum(row["stage"] == "request_relational_tie" for row in episodes), 2)


if __name__ == "__main__":
    unittest.main()
