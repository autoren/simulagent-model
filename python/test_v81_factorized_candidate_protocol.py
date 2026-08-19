#!/usr/bin/env python3
from __future__ import annotations

import json
import unittest

from v22r2_grounding import PROJECT_ROOT
from v81_factorized_candidate_protocol import (
    aggregate,
    compose_candidates,
    evaluate_gates,
    parse_factorized_response,
    score_record,
)


class V81ProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        lock = json.loads(
            (PROJECT_ROOT / "configs/v81-factorized-local-candidate-design-lock.json").read_text()
        )
        cls.config = lock["config_payload"]
        cls.records = cls.config["records"]

    def response(self, labels: dict) -> str:
        return json.dumps(labels)

    def ordered_labels(self, labels: dict) -> dict:
        return {key: labels[key] for key in self.config["labelKeysInRequiredOrder"]}

    def test_composer_has_locked_canonical_order_and_escape(self) -> None:
        labels = {
            "schedule_review": True,
            "send_summary": True,
            "alex_chen": True,
            "alex_kim": True,
            "out_of_ontology": False,
        }
        self.assertEqual(
            compose_candidates(labels), self.config["candidateIdsInRequiredOrder"]
        )
        labels.update(
            schedule_review=False, send_summary=False, out_of_ontology=True
        )
        self.assertEqual(compose_candidates(labels), ["none_of_the_above"])

    def test_strict_schema_rejects_extra_fields_nonbooleans_and_inconsistency(self) -> None:
        labels = self.ordered_labels(self.records[0]["goldLabels"])
        self.assertTrue(
            parse_factorized_response(self.response(labels), self.config)["schema_valid"]
        )
        extra = {**labels, "confidence": 0.9}
        self.assertFalse(
            parse_factorized_response(self.response(extra), self.config)["schema_valid"]
        )
        nonboolean = {**labels, "schedule_review": 1}
        self.assertFalse(
            parse_factorized_response(self.response(nonboolean), self.config)["schema_valid"]
        )
        inconsistent = {**labels, "out_of_ontology": True}
        self.assertFalse(
            parse_factorized_response(self.response(inconsistent), self.config)["schema_valid"]
        )

    def test_perfect_population_passes_all_gates(self) -> None:
        rows = [
            score_record(
                record,
                self.response(self.ordered_labels(record["goldLabels"])),
                self.config,
            )
            for record in self.records
        ]
        access = {
            "model_generation_count": 24,
            "API_call_count": 0,
            "adapter_training_run_count": 0,
            "human_record_access_count": 0,
            "real_tool_call_count": 0,
            "external_side_effect_count": 0,
        }
        self.assertTrue(all(evaluate_gates(aggregate(rows), self.config, access).values()))

    def test_candidate_fields_and_twenty_fifth_generation_fail(self) -> None:
        labels = self.ordered_labels(self.records[0]["goldLabels"])
        value = {**labels, "candidate_ids": ["none_of_the_above"]}
        parsed = parse_factorized_response(self.response(value), self.config)
        self.assertFalse(parsed["schema_valid"])
        self.assertEqual(parsed["candidate_id_field_count"], 1)
        rows = [
            score_record(
                record,
                self.response(self.ordered_labels(record["goldLabels"])),
                self.config,
            )
            for record in self.records
        ]
        access = {
            "model_generation_count": 25,
            "API_call_count": 0,
            "adapter_training_run_count": 0,
            "human_record_access_count": 0,
            "real_tool_call_count": 0,
            "external_side_effect_count": 0,
        }
        self.assertFalse(
            evaluate_gates(aggregate(rows), self.config, access)[
                "bounded_local_model_and_zero_external_access"
            ]
        )


if __name__ == "__main__":
    unittest.main()
