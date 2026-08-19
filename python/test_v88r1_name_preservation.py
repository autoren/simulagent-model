#!/usr/bin/env python3
from __future__ import annotations

import json
import unittest

from run_v88r1_external_candidate_mlx import score_named_record


class V88R1NamePreservationTests(unittest.TestCase):
    def test_only_registered_name_is_copied_into_scored_row(self):
        record = {
            "name": "registered-name",
            "id": "fixture-id",
            "source_record_id": "source-id",
            "service": "Service_1",
            "allowed_intent_ids": ["Book", "NONE"],
            "allowed_slot_ids": ["city"],
            "gold": {
                "active_intent": "Book",
                "intent_candidates": ["Book", "NONE"],
                "state_slot_key_candidates": ["city"],
            },
        }
        response = json.dumps({
            "intent_candidates": ["Book", "NONE"],
            "state_slot_key_candidates": ["city"],
        })
        row = score_named_record(record, response)
        self.assertEqual(row["name"], "registered-name")
        self.assertEqual(row["id"], "fixture-id")
        self.assertTrue(row["intent_candidate_exact"])
        self.assertTrue(row["state_slot_key_exact"])


if __name__ == "__main__":
    unittest.main()
