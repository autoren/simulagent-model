import json
import unittest
from pathlib import Path

from v135_controlled_open_world_minimal_pairs import build_catalog, build_population
from v139_repaired_direct_vs_thinking_realization import (
    evaluate_condition,
    render_prompt,
    validate_final_answer_v138,
)


class V139RepairedDirectVsThinkingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = json.loads(Path("configs/v139-repaired-direct-vs-thinking-realization.json").read_text())
        cls.v135 = json.loads(Path("configs/v135-controlled-open-world-minimal-pairs.json").read_text())
        cls.v136 = json.loads(Path("configs/v136-controlled-clarification-value.json").read_text())
        cls.catalog = build_catalog(cls.v135)
        population = build_population(cls.v135)
        cls.public = [row for row in population["public_fixtures"] if row["split"] == "development"]
        cls.hidden = [row for row in population["hidden_fixtures"] if row["split"] == "development"]

    def test_population_is_exact_unused_development_split(self):
        self.assertEqual(len(self.public), 100)
        self.assertEqual(len(self.hidden), 100)
        self.assertEqual(len({row["group_id"] for row in self.hidden}), 20)

    def test_prompt_contains_no_hidden_labels(self):
        prompt = render_prompt(self.catalog, self.public[0], self.config)
        self.assertIn("conversation", prompt)
        self.assertNotIn("truth_choice_id", prompt)
        self.assertNotIn("group_id", prompt)

    def test_repaired_thinking_and_direct_parsing(self):
        thinking = validate_final_answer_v138(
            'compare\n</think>\n{"choice_id":"N01"}',
            self.catalog,
            thinking_enabled=True,
            prompt_think_opened=True,
        )
        direct = validate_final_answer_v138(
            '{"choice_id":"K01"}',
            self.catalog,
            thinking_enabled=False,
            prompt_think_opened=False,
        )
        self.assertTrue(thinking["response_valid"])
        self.assertTrue(direct["response_valid"])

    def test_perfect_mock_outputs_qualify_both_conditions(self):
        for condition_id, trace in (("direct", False), ("thinking", True)):
            outputs = {}
            for row in self.hidden:
                name = f"{condition_id}::{row['fixture_id']}"
                outputs[name] = {
                    "name": name,
                    "condition_id": condition_id,
                    "fixture_id": row["fixture_id"],
                    "answer_choice_id": row["truth_choice_id"],
                    "response_valid": True,
                    "thinking_trace_present": trace,
                    "generated_token_count": 8,
                    "generation_seconds": 0.01,
                }
            outcome = evaluate_condition(
                condition_id, outputs, self.hidden, self.catalog, self.v136, self.config
            )
            self.assertTrue(outcome["qualified"], outcome)


if __name__ == "__main__":
    unittest.main()
