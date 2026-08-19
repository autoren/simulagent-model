#!/usr/bin/env python3
from __future__ import annotations

import json
import unittest

from v88_external_candidate_protocol import aggregate, control_rows, format_user_prompt, parse_candidate_response, score_response


def fixture(active_intent="Book", slots=None):
    slots = ["city"] if slots is None else slots
    return {
        "id": "fixture",
        "source_record_id": "dialogue::turn-001::Service_1",
        "service": "Service_1",
        "dialogue_history": [{"speaker": "USER", "utterance": "human words"}],
        "schema_context": {
            "service_name": "Service_1",
            "service_description": "service description",
            "intents": [{"id": "Book", "description": "book something"}],
            "slots": [{"id": "city", "description": "target city"}],
        },
        "allowed_intent_ids": ["Book", "NONE"],
        "allowed_slot_ids": ["city"],
        "gold": {
            "active_intent": active_intent,
            "intent_candidates": ["NONE"] if active_intent == "NONE" else ["Book", "NONE"],
            "state_slot_key_candidates": slots,
        },
    }


class V88ExternalCandidateProtocolTests(unittest.TestCase):
    def test_prompt_uses_frozen_template_without_target_leak(self):
        config = {"userPromptTemplate": "{service_name}\n{service_description}\n{intent_lines}\n{slot_lines}\n{dialogue_history}"}
        prompt = format_user_prompt(fixture(), config)
        self.assertIn("human words", prompt)
        self.assertIn("Book: book something", prompt)
        self.assertNotIn("intent_candidates", prompt)

    def test_exact_valid_response_scores_perfectly(self):
        response = json.dumps({"intent_candidates": ["Book", "NONE"], "state_slot_key_candidates": ["city"]})
        row = score_response(fixture(), response)
        self.assertTrue(row["exact_json"])
        self.assertTrue(row["ontology_conformant"])
        self.assertTrue(row["mandatory_NONE_included"])
        self.assertTrue(row["intent_candidate_exact"])
        self.assertTrue(row["state_slot_key_exact"])
        self.assertFalse(row["executable"])

    def test_extra_key_duplicate_and_out_of_ontology_fail_closed(self):
        cases = [
            {"intent_candidates": ["Book", "NONE"], "state_slot_key_candidates": ["city"], "action": "book"},
            {"intent_candidates": ["NONE", "NONE"], "state_slot_key_candidates": []},
            {"intent_candidates": ["Delete", "NONE"], "state_slot_key_candidates": []},
        ]
        for case in cases:
            parsed = parse_candidate_response(json.dumps(case), fixture())
            self.assertFalse(parsed["ontology_conformant"])

    def test_genuine_NONE_requires_NONE_only(self):
        exact = score_response(fixture(active_intent="NONE", slots=[]), json.dumps({"intent_candidates": ["NONE"], "state_slot_key_candidates": []}))
        broad = score_response(fixture(active_intent="NONE", slots=[]), json.dumps({"intent_candidates": ["Book", "NONE"], "state_slot_key_candidates": []}))
        self.assertTrue(exact["intent_candidate_exact"])
        self.assertFalse(broad["intent_candidate_exact"])

    def test_controls_and_aggregate_expose_enumeration_shortcut(self):
        record = fixture()
        row = score_response(record, json.dumps({"intent_candidates": ["Book", "NONE"], "state_slot_key_candidates": ["city"]}))
        controls = control_rows(record)
        self.assertTrue(controls["exhaustive"]["intent_exact"])
        metrics = aggregate([row], {record["id"]: record})
        self.assertEqual(metrics["intent_candidate_set_exact_rate"], 1.0)
        self.assertEqual(metrics["state_slot_key_exact_rate"], 1.0)


if __name__ == "__main__":
    unittest.main()
