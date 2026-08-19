import json
import unittest
from pathlib import Path

from v156_model_free_explicit_metadata_question_retrieval import (
    build_episodes,
    normalize_tokens,
    rank_queries,
    score_query,
)


class V156ModelFreeExplicitMetadataQuestionRetrievalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = json.loads(
            Path("configs/v156-model-free-explicit-metadata-question-retrieval.json").read_text()
        )
        cls.catalog = {
            "queries": [
                {
                    "query_id": "Q1", "title": "identify_garden_permit",
                    "question": "Which permit?",
                    "retrieval_profile": {
                        "anchor_phrases": ["community garden"],
                        "primary_terms": ["garden", "permit"], "secondary_terms": ["renew"],
                    },
                    "options": [{"option_id": "L", "text": "Garden permit"}, {"option_id": "R", "text": "Other permit"}],
                },
                {
                    "query_id": "Q2", "title": "identify_device",
                    "question": "Which device?",
                    "retrieval_profile": {
                        "anchor_phrases": ["marine buoy"],
                        "primary_terms": ["marine", "buoy"], "secondary_terms": ["calibrate"],
                    },
                    "options": [{"option_id": "L", "text": "Marine device"}, {"option_id": "R", "text": "Other device"}],
                },
            ]
        }

    def test_normalization_is_deterministic_and_punctuation_insensitive(self):
        self.assertEqual(normalize_tokens("Café—GARDEN permit #17"), ["cafe", "garden", "permit", "17"])

    def test_anchor_primary_and_secondary_weights_are_exact(self):
        query = self.catalog["queries"][0]
        scored = score_query("Renew the community garden permit", query, self.config)
        expected = 8.0 + 2 * 3.0 + 1.0 + 2 * 0.25
        self.assertEqual(scored["score"], expected)
        self.assertEqual(scored["anchor_phrase_hit_count"], 1)

    def test_ranking_uses_only_visible_text_and_state_free_catalog(self):
        fixture = {"conversation": [{"role": "user", "text": "Calibrate marine buoy B-7"}]}
        ranked = rank_queries(fixture, self.catalog, self.config)
        self.assertEqual(ranked["query_ranking"], ["Q2", "Q1"])
        self.assertGreater(ranked["top_two_margin"], 0)
        self.assertNotIn("state_id", json.dumps(self.catalog))
        self.assertNotIn("witness", json.dumps(self.catalog))

    def test_equal_scores_use_source_order_and_report_tie(self):
        fixture = {"conversation": [{"role": "user", "text": "Please help"}]}
        ranked = rank_queries(fixture, self.catalog, self.config)
        self.assertEqual(ranked["query_ranking"], ["Q1", "Q2"])
        self.assertEqual(ranked["top_score_tie_count"], 2)

    def test_episode_builder_expands_ambiguous_request_to_both_sides(self):
        rows = [
            {"fixture_id": "r", "group_id": "g", "stage": "request_ambiguous", "oracle_query_id": "Q1", "truth_state_id": "A00", "closed_answer_event": None},
            {"fixture_id": "l", "group_id": "g", "stage": "closed_answer_left", "oracle_query_id": "Q1", "truth_state_id": "K1", "closed_answer_event": {"query_id": "Q1", "selected_option_id": "LEFT"}},
            {"fixture_id": "x", "group_id": "g", "stage": "closed_answer_right", "oracle_query_id": "Q1", "truth_state_id": "N1", "closed_answer_event": {"query_id": "Q1", "selected_option_id": "RIGHT"}},
        ]
        episodes = build_episodes(rows)
        self.assertEqual(len(episodes), 2)
        self.assertEqual({row["side"] for row in episodes}, {"left", "right"})


if __name__ == "__main__":
    unittest.main()
