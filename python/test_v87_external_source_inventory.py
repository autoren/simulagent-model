#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import unittest

from v87_external_source_inventory import build_structural_inventory, compile_schema_index, git_blob_sha1


SCHEMA = [
    {
        "service_name": "Flights_1",
        "slots": [{"name": "destination"}],
        "intents": [{"name": "SearchFlight"}],
    },
    {
        "service_name": "Calendar_1",
        "slots": [{"name": "date"}],
        "intents": [{"name": "CreateEvent"}],
    },
]


def dialogue_payload(active_intent: str = "SearchFlight", slot: str = "destination"):
    return [
        {
            "dialogue_id": "dlg-1",
            "turns": [
                {"speaker": "SYSTEM", "utterance": "system language", "frames": []},
                {
                    "speaker": "USER",
                    "utterance": "private human language",
                    "frames": [
                        {
                            "service": "Flights_1",
                            "state": {
                                "active_intent": active_intent,
                                "requested_slots": [],
                                "slot_values": {slot: ["private slot value"]},
                            },
                        }
                    ],
                },
                {
                    "speaker": "USER",
                    "utterance": "excluded language",
                    "frames": [
                        {
                            "service": "Calendar_1",
                            "state": {
                                "active_intent": "CreateEvent",
                                "requested_slots": [],
                                "slot_values": {"date": ["tomorrow"]},
                            },
                        }
                    ],
                },
            ],
        }
    ]


class V87ExternalSourceInventoryTests(unittest.TestCase):
    def test_git_blob_identity_matches_git_definition(self):
        data = b"hello\n"
        expected = hashlib.sha1(b"blob 6\0hello\n").hexdigest()  # noqa: S324
        self.assertEqual(git_blob_sha1(data), expected)

    def test_schema_compiler_requires_unique_typed_identifiers(self):
        index = compile_schema_index(SCHEMA)
        self.assertEqual(index["Flights_1"]["slot_names"], frozenset({"destination"}))
        bad = [{"service_name": "X_1", "slots": [{"name": "a"}, {"name": "a"}], "intents": []}]
        with self.assertRaisesRegex(ValueError, "duplicate"):
            compile_schema_index(bad)

    def test_inventory_is_structural_and_excludes_registered_prefix(self):
        result = build_structural_inventory(
            SCHEMA, dialogue_payload(), excluded_service_prefixes=("Calendar_",)
        )
        self.assertEqual(result["counts"]["eligible_record_count"], 1)
        self.assertEqual(result["counts"]["eligible_active_record_count"], 1)
        self.assertEqual(result["ineligibility_reason_counts"]["excluded_service_prefix"], 1)
        self.assertFalse(result["contains_utterance_or_text_fields"])
        serialized = str(result)
        self.assertNotIn("private human language", serialized)
        self.assertNotIn("private slot value", serialized)
        self.assertNotIn("excluded language", serialized)

    def test_none_label_is_retained_as_open_set_control(self):
        result = build_structural_inventory(
            SCHEMA, dialogue_payload(active_intent="NONE"), excluded_service_prefixes=("Calendar_",)
        )
        self.assertEqual(result["counts"]["eligible_none_record_count"], 1)
        self.assertEqual(result["record_index"][0]["label_kind"], "none")

    def test_invalid_schema_linkage_is_ineligible(self):
        bad_intent = build_structural_inventory(
            SCHEMA,
            dialogue_payload(active_intent="UnknownIntent"),
            excluded_service_prefixes=("Calendar_",),
        )
        self.assertEqual(bad_intent["ineligibility_reason_counts"]["active_intent_not_in_schema"], 1)
        self.assertNotIn("eligible_record_count", bad_intent["counts"])
        bad_slot = build_structural_inventory(
            SCHEMA,
            dialogue_payload(slot="unknown_slot"),
            excluded_service_prefixes=("Calendar_",),
        )
        self.assertEqual(bad_slot["ineligibility_reason_counts"]["slot_not_in_schema"], 1)
        self.assertNotIn("eligible_record_count", bad_slot["counts"])


if __name__ == "__main__":
    unittest.main()
