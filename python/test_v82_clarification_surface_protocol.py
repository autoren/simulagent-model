#!/usr/bin/env python3
from __future__ import annotations

import json
import unittest

from v22r2_grounding import PROJECT_ROOT
from v82_clarification_surface_protocol import (
    aggregate,
    control_metrics,
    evaluate_gates,
    grammar_surface,
    parse_and_render,
    validate_surface,
)


class V82SurfaceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        lock = json.loads(
            (PROJECT_ROOT / "configs/v82-local-clarification-surface-design-lock.json").read_text()
        )
        cls.config = lock["config_payload"]
        cls.records = cls.config["records"]

    def test_canonical_and_finite_grammar_surfaces_validate(self) -> None:
        for record in self.records:
            code = record["clarificationCode"]
            self.assertTrue(
                validate_surface(code, self.config["canonicalSurfaces"][code], self.config)[
                    "semantic_valid"
                ]
            )
            self.assertTrue(
                validate_surface(
                    code, grammar_surface(code, record["styleHint"], self.config), self.config
                )["semantic_valid"]
            )

    def test_valid_raw_surface_is_preserved(self) -> None:
        record = self.records[0]
        question = grammar_surface(
            record["clarificationCode"], "polite", self.config
        )
        row = parse_and_render(record, json.dumps({"question": question}), self.config)
        self.assertTrue(row["raw_semantic_valid"])
        self.assertFalse(row["fallback_used"])
        self.assertEqual(row["final_question"], question)
        self.assertEqual(row["resolved_action_code"], record["clarificationCode"])

    def test_invalid_output_is_discarded_for_canonical_fallback(self) -> None:
        record = self.records[0]
        response = json.dumps({"question": "I will schedule the project review or send the project summary?"})
        row = parse_and_render(record, response, self.config)
        self.assertFalse(row["raw_semantic_valid"])
        self.assertTrue(row["fallback_used"])
        self.assertTrue(row["final_semantic_valid"])
        self.assertEqual(
            row["final_question"],
            self.config["canonicalSurfaces"][record["clarificationCode"]],
        )

    def test_unsafe_mutations_are_all_rejected(self) -> None:
        controls = control_metrics(self.config)
        self.assertEqual(controls["canonical_baseline_validity_rate"], 1.0)
        self.assertEqual(controls["finite_grammar_baseline_validity_rate"], 1.0)
        self.assertEqual(controls["unsafe_mutation_rejection_rate"], 1.0)

    def test_perfect_grammar_population_passes_all_gates(self) -> None:
        rows = [
            parse_and_render(
                record,
                json.dumps(
                    {
                        "question": grammar_surface(
                            record["clarificationCode"], record["styleHint"], self.config
                        )
                    }
                ),
                self.config,
            )
            for record in self.records
        ]
        policy = {
            "reachable_clarification_action_invariance_rate": 1.0,
            "maximum_policy_value_absolute_error": 0.0,
        }
        access = {
            "model_generation_count": 24,
            "API_call_count": 0,
            "adapter_training_run_count": 0,
            "human_record_access_count": 0,
            "original_user_language_access_count": 0,
            "real_tool_call_count": 0,
            "external_side_effect_count": 0,
        }
        self.assertTrue(
            all(
                evaluate_gates(
                    aggregate(rows), control_metrics(self.config), policy, self.config, access
                ).values()
            )
        )


if __name__ == "__main__":
    unittest.main()
