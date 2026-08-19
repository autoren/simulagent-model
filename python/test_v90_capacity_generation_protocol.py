#!/usr/bin/env python3
from __future__ import annotations

import json
import unittest

from v90_capacity_generation_protocol import (
    aggregate,
    paired_correctness_transitions,
    quality_gate_pass,
    score_response,
    union_scored_row,
)


def fixture(active="Book", slots=None):
    slots = ["city"] if slots is None else slots
    return {
        "id": "fixture",
        "source_record_id": "d::turn-001::Service_1",
        "service": "Service_1",
        "allowed_intent_ids": ["Book", "Cancel", "NONE"],
        "allowed_slot_ids": ["city", "date"],
        "gold": {
            "active_intent": active,
            "intent_candidates": ["NONE"] if active == "NONE" else [active, "NONE"],
            "state_slot_key_candidates": slots,
        },
    }


def scored(record, intents, slots):
    row = score_response(record, json.dumps({
        "intent_candidates": intents,
        "state_slot_key_candidates": slots,
    }))
    row["name"] = record["id"]
    return row


class V90ProtocolTests(unittest.TestCase):
    def test_union_retains_complementary_valid_target_and_none(self):
        record = fixture()
        left = scored(record, ["NONE"], ["city"])
        right = scored(record, ["Book", "NONE"], ["date"])
        row = union_scored_row(record, left, right)
        self.assertTrue(row["ontology_conformant"])
        self.assertTrue(row["gold_active_intent_covered"])
        self.assertEqual(set(row["intent_candidates"]), {"Book", "NONE"})
        self.assertEqual(set(row["state_slot_key_candidates"]), {"city", "date"})
        self.assertFalse(row["executable"])

    def test_union_fails_closed_if_either_model_is_nonconforming(self):
        record = fixture()
        left = scored(record, ["Book", "NONE"], ["city"])
        right = score_response(record, "not-json")
        row = union_scored_row(record, left, right)
        self.assertFalse(row["ontology_conformant"])
        self.assertEqual(row["intent_candidates"], [])
        self.assertFalse(row["executable"])

    def test_paired_transitions_are_identity_checked_and_exact(self):
        record = fixture()
        left = {"fixture": scored(record, ["NONE"], [])}
        right = {"fixture": scored(record, ["Book", "NONE"], ["city"])}
        transitions = paired_correctness_transitions(left, right)
        self.assertEqual(transitions["active_intent_covered"]["false_to_true"], 1)
        self.assertEqual(transitions["state_exact"]["false_to_true"], 1)
        with self.assertRaises(ValueError):
            paired_correctness_transitions(left, {})

    def test_quality_gate_pass_ignores_only_access_keys(self):
        from v90_capacity_generation_protocol import QUALITY_GATE_KEYS
        gates = {key: True for key in QUALITY_GATE_KEYS}
        gates["model_load_budget"] = False
        self.assertTrue(quality_gate_pass(gates))
        gates["state_slot_key_exact"] = False
        self.assertFalse(quality_gate_pass(gates))

    def test_aggregate_remains_identical_to_frozen_v88_semantics(self):
        record = fixture()
        row = scored(record, ["Book", "NONE"], ["city"])
        metrics = aggregate([row], {record["id"]: record})
        self.assertEqual(metrics["intent_candidate_set_exact_rate"], 1.0)
        self.assertEqual(metrics["state_slot_key_exact_rate"], 1.0)


if __name__ == "__main__":
    unittest.main()
