#!/usr/bin/env python3
"""Outcome-blind parser, scorer, and mutation tests for V80."""
from __future__ import annotations

import json
import unittest

from v22r2_grounding import PROJECT_ROOT
from v80_candidate_protocol import aggregate, evaluate_gates, score_record


class V80CandidateProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        lock = json.loads(
            (PROJECT_ROOT / "configs/v80-local-candidate-generation-design-lock.json").read_text()
        )
        cls.config = lock["config_payload"]

    def test_exact_canonical_response_passes(self) -> None:
        record = self.config["records"][0]
        response = json.dumps({"candidate_ids": record["goldCandidateIds"]})
        row = score_record(record, response, self.config)
        self.assertTrue(row["exact_json_parse"])
        self.assertTrue(row["schema_valid"])
        self.assertTrue(row["exact_candidate_set"])

    def test_fence_explanation_extra_field_duplicate_and_order_fail(self) -> None:
        record = self.config["records"][0]
        cases = (
            "```json\n{}\n```",
            json.dumps({"candidate_ids": record["goldCandidateIds"], "confidence": 1.0}),
            json.dumps({"candidate_ids": ["none_of_the_above", record["goldCandidateIds"][0]]}),
            json.dumps({"candidate_ids": [record["goldCandidateIds"][0]] * 2 + ["none_of_the_above"]}),
        )
        self.assertTrue(
            all(not score_record(record, response, self.config)["schema_valid"] for response in cases)
        )

    def test_forbidden_fields_are_counted(self) -> None:
        record = self.config["records"][0]
        row = score_record(
            record,
            json.dumps(
                {
                    "candidate_ids": record["goldCandidateIds"],
                    "probability": 0.9,
                    "tool_call": "send",
                }
            ),
            self.config,
        )
        self.assertEqual(row["confidence_or_probability_field_count"], 1)
        self.assertEqual(row["action_or_tool_field_count"], 1)

    def test_perfect_synthetic_responses_pass_full_gate_path(self) -> None:
        rows = [
            score_record(
                record,
                json.dumps({"candidate_ids": record["goldCandidateIds"]}),
                self.config,
            )
            for record in self.config["records"]
        ]
        metrics = aggregate(rows)
        access = {
            "model_forward_pass_count": 24,
            "API_call_count": 0,
            "adapter_training_run_count": 0,
            "human_record_access_count": 0,
            "real_tool_call_count": 0,
            "external_side_effect_count": 0,
        }
        gates = evaluate_gates(metrics, self.config, access)
        self.assertTrue(all(isinstance(value, bool) and value for value in gates.values()))


if __name__ == "__main__":
    unittest.main()
