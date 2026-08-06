"""Regression tests for strict transition-output validation."""

from __future__ import annotations

import json
import unittest
from pathlib import Path

from evaluate_predictions import valid_schema
from evaluate_outcome_count import evaluate_counts, normalize_count, valid_count_schema
from predict_mlx import SYSTEM_PROMPT
from predict_epistemic_mlx import SYSTEM_PROMPT as EPISTEMIC_SYSTEM_PROMPT
from predict_outcome_count_mlx import SYSTEM_PROMPT as OUTCOME_COUNT_SYSTEM_PROMPT


class TransitionSchemaTests(unittest.TestCase):
    def setUp(self) -> None:
        self.valid = {
            "blocked_actions_added": [],
            "blocked_actions_removed": [],
            "environment_changed": False,
            "flags_changed": {},
            "hidden_actions_concealed": [],
            "hidden_actions_revealed": [],
            "inventory_added": [],
            "inventory_removed": [],
            "next_location": "atrium",
            "reachable_room_delta": 0,
            "success": True,
            "visible_actions_added": [],
            "visible_actions_removed": [],
        }

    def test_accepts_exact_typed_schema(self) -> None:
        self.assertTrue(valid_schema(self.valid))

    def test_rejects_missing_field(self) -> None:
        incomplete = dict(self.valid)
        incomplete.pop("visible_actions_removed")
        self.assertFalse(valid_schema(incomplete))

    def test_rejects_wrong_field_type(self) -> None:
        invalid = dict(self.valid)
        invalid["success"] = "true"
        self.assertFalse(valid_schema(invalid))

    def test_rejects_non_string_array_items(self) -> None:
        invalid = dict(self.valid)
        invalid["inventory_added"] = [1]
        self.assertFalse(valid_schema(invalid))

    def test_rejects_non_boolean_flag_values(self) -> None:
        invalid = dict(self.valid)
        invalid["flags_changed"] = {"doorUnlocked": "true"}
        self.assertFalse(valid_schema(invalid))

    def test_rejects_boolean_room_delta(self) -> None:
        invalid = dict(self.valid)
        invalid["reachable_room_delta"] = False
        self.assertFalse(valid_schema(invalid))

    def test_training_and_inference_prompts_match(self) -> None:
        path = Path("data/pilot/mlx/agent/train.jsonl")
        row = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
        self.assertEqual(row["messages"][0]["content"], SYSTEM_PROMPT)

    def test_v2_training_and_inference_prompts_match(self) -> None:
        path = Path("data/v2/mlx/agent/train.jsonl")
        row = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
        self.assertEqual(row["messages"][0]["content"], EPISTEMIC_SYSTEM_PROMPT)

    def test_outcome_count_training_and_inference_prompts_match(self) -> None:
        path = Path("data/v2/mlx/outcome-count/train.jsonl")
        row = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
        self.assertEqual(row["messages"][0]["content"], OUTCOME_COUNT_SYSTEM_PROMPT)

    def test_outcome_count_schema_is_bounded_integer(self) -> None:
        self.assertTrue(valid_count_schema({"outcome_count": 3}))
        self.assertFalse(valid_count_schema({"outcome_count": False}))
        self.assertFalse(valid_count_schema({"outcome_count": 6}))
        self.assertEqual(normalize_count("3"), 3)
        self.assertIsNone(normalize_count("count: 3"))

    def test_outcome_count_evaluation(self) -> None:
        gold = [
            {
                "id": "two",
                "target": {"possible_outcomes": [self.valid, dict(self.valid, success=False)]},
            }
        ]
        report = evaluate_counts(
            gold,
            [{"id": "two", "prediction": "2"}],
        )
        self.assertEqual(report["accuracy"], 1.0)
        self.assertEqual(report["mean_absolute_error_given_valid_prediction"], 0.0)


if __name__ == "__main__":
    unittest.main()
