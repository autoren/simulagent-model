import json
import unittest
from pathlib import Path

from v135_controlled_open_world_minimal_pairs import build_catalog, build_population
from v137_direct_vs_thinking_realization import evaluate_condition, render_prompt, validate_final_answer


class V137DirectVsThinkingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = json.loads(Path("configs/v137-direct-vs-thinking-realization.json").read_text())
        cls.v135 = json.loads(Path("configs/v135-controlled-open-world-minimal-pairs.json").read_text())
        cls.v136 = json.loads(Path("configs/v136-controlled-clarification-value.json").read_text())
        cls.catalog = build_catalog(cls.v135)
        population = build_population(cls.v135)
        cls.public = [row for row in population["public_fixtures"] if row["split"] == "test"]
        cls.hidden = [row for row in population["hidden_fixtures"] if row["split"] == "test"]

    def test_direct_and_thinking_final_parsing(self):
        direct = validate_final_answer('{"choice_id":"K01"}', self.catalog, False)
        self.assertTrue(direct["response_valid"])
        self.assertFalse(direct["thinking_trace_present"])
        thought = validate_final_answer('<think>compare boundaries</think>\n{"choice_id":"N01"}', self.catalog, True)
        self.assertTrue(thought["response_valid"])
        self.assertTrue(thought["thinking_trace_present"])
        self.assertEqual(thought["answer_choice_id"], "N01")

    def test_unclosed_or_extra_direct_reasoning_is_invalid(self):
        self.assertFalse(validate_final_answer('<think>unfinished', self.catalog, True)["response_valid"])
        self.assertFalse(validate_final_answer('reasoning\n{"choice_id":"K01"}', self.catalog, False)["response_valid"])

    def test_prompt_contains_catalog_candidate_and_conversation_but_no_gold(self):
        prompt = render_prompt(self.catalog, self.public[0], self.config)
        self.assertIn("presented_candidate_under_review", prompt)
        self.assertIn("conversation", prompt)
        self.assertNotIn("truth_choice_id", prompt)
        self.assertNotIn("group_id", prompt)

    def test_perfect_mock_outputs_qualify_both_conditions(self):
        for condition_id, trace in (("direct", False), ("thinking", True)):
            outputs = {}
            by_hidden = {row["fixture_id"]: row for row in self.hidden}
            for fixture_id, row in by_hidden.items():
                name = f"{condition_id}::{fixture_id}"
                outputs[name] = {
                    "name": name,
                    "condition_id": condition_id,
                    "fixture_id": fixture_id,
                    "answer_choice_id": row["truth_choice_id"],
                    "response_valid": True,
                    "thinking_trace_present": trace,
                    "generated_token_count": 8,
                    "generation_seconds": 0.01,
                }
            result = evaluate_condition(condition_id, outputs, self.hidden, self.catalog, self.v136, self.config)
            self.assertTrue(result["qualified"], result)
            self.assertAlmostEqual(result["metrics"]["sequential_mean_decision_cost"], 0.3)


if __name__ == "__main__":
    unittest.main()
