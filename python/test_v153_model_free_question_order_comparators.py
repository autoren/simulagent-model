import json
import unittest
from pathlib import Path

from v153_model_free_question_order_comparators import COMPARATORS, build_episodes, comparator_order, evaluate


class V153ModelFreeQuestionOrderComparatorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = json.loads(Path("configs/v153-model-free-question-order-comparators.json").read_text())
        cls.witness_config = json.loads(Path("configs/v152-fresh-question-order-population.json").read_text())
        design = Path("outputs/v152-fresh-question-order-population/design")
        allowed = {
            "fixture_id", "split", "group_id", "family_id", "stage", "truth_state_id",
            "oracle_query_id", "closed_answer_event",
        }
        cls.hidden = [
            {key: row[key] for key in allowed}
            for row in json.loads((design / "hidden-fixtures.json").read_text())
            if row["split"] == "development"
        ]
        cls.catalog = json.loads((design / "interaction-catalog.json").read_text())
        cls.result = evaluate(cls.hidden, cls.catalog, cls.witness_config, cls.config)

    def test_episode_count_and_request_expansion(self):
        episodes = build_episodes(self.hidden)
        self.assertEqual(len(episodes), 120)
        self.assertEqual(len({row["fixture_id"] for row in episodes}), 96)
        self.assertTrue(all("conversation" not in row for row in self.hidden))

    def test_orders_are_complete_deterministic_and_oracle_first(self):
        query_ids = [row["query_id"] for row in self.catalog["queries"]]
        for episode in build_episodes(self.hidden):
            for comparator in COMPARATORS:
                first = comparator_order(comparator, episode, query_ids, self.config)
                second = comparator_order(comparator, episode, query_ids, self.config)
                self.assertEqual(first, second)
                if comparator == "NO_QUERY":
                    self.assertEqual(first, [])
                else:
                    self.assertEqual(set(first), set(query_ids))
                    self.assertEqual(len(first), len(query_ids))
            self.assertEqual(
                comparator_order("ORACLE_ORDER", episode, query_ids, self.config)[0],
                episode["oracle_query_id"],
            )

    def test_reference_ranks_and_costs(self):
        metrics = self.result["metrics"]
        self.assertEqual(metrics["ORACLE_ORDER"]["mean_correct_query_rank"], 1.0)
        self.assertAlmostEqual(metrics["ORACLE_ORDER"]["mean_decision_cost"], 0.3)
        self.assertEqual(metrics["SOURCE_ORDER"]["mean_correct_query_rank"], 3.5)
        self.assertEqual(metrics["NO_QUERY"]["mean_decision_cost"], 1.0)

    def test_trusted_answer_policies_are_exact_and_fail_closed_between_queries(self):
        for comparator in ("SOURCE_ORDER", "SEEDED_RANDOM", "ORACLE_ORDER"):
            row = self.result["metrics"][comparator]
            self.assertEqual(row["final_exact_accuracy"], 1.0)
            self.assertEqual(row["irrelevant_query_fail_closed_rate"], 1.0)
            self.assertEqual(row["authoritative_hypothesis_retention"], 1.0)
            self.assertEqual(row["actual_execution_count"], 0)

    def test_no_candidate_surface_or_prohibited_access(self):
        self.assertEqual(self.result["candidate_proposal_field_count"], 0)
        for key in (
            "evaluation_language_read_count", "model_load_count", "model_generation_or_score_count",
            "API_call_count", "training_run_count", "actual_execution_count",
        ):
            self.assertEqual(self.result[key], 0)

    def test_all_gates_pass(self):
        self.assertTrue(self.result["passed"], self.result)


if __name__ == "__main__":
    unittest.main()
