from __future__ import annotations

import hashlib
import unittest

from v93_open_set_source import (
    build_open_set_inventory,
    compile_schema,
    git_blob_sha1,
    normalized_tokens,
)


SCHEMA = [
    {
        "service_name": "Alpha_1",
        "description": "alpha",
        "slots": [{"name": "item", "description": "item", "is_categorical": False, "possible_values": []}],
        "intents": [
            {"name": "AddItem", "description": "add an item"},
            {"name": "RemoveItem", "description": "remove an item"},
            {"name": "InspectItem", "description": "inspect an item"},
            {"name": "UpdateItem", "description": "update an item"},
        ],
    },
    {
        "service_name": "Beta_1",
        "description": "beta",
        "slots": [{"name": "place", "description": "place", "is_categorical": False, "possible_values": []}],
        "intents": [
            {"name": "BookPlace", "description": "book a place"},
            {"name": "CancelPlace", "description": "cancel a place"},
            {"name": "FindPlace", "description": "find a place"},
            {"name": "RatePlace", "description": "rate a place"},
        ],
    },
]


def frame(service: str, intent: str, slot: str) -> dict:
    return {
        "service": service,
        "state": {"active_intent": intent, "requested_slots": [], "slot_values": {slot: ["private"]}},
    }


def dialogues() -> list[dict]:
    rows = []
    for index, (service, intent, utterance, slot) in enumerate(
        [
            ("Alpha_1", "AddItem", "add the object", "item"),
            ("Alpha_1", "RemoveItem", "discard the object", "item"),
            ("Alpha_1", "InspectItem", "inspect the object", "item"),
            ("Alpha_1", "UpdateItem", "revise the object", "item"),
            ("Alpha_1", "NONE", "hmm", "item"),
            ("Beta_1", "BookPlace", "book somewhere", "place"),
            ("Beta_1", "CancelPlace", "forget my reservation", "place"),
            ("Beta_1", "FindPlace", "find somewhere", "place"),
            ("Beta_1", "RatePlace", "score my destination", "place"),
            ("Beta_1", "NONE", "not sure", "place"),
        ]
    ):
        rows.append(
            {
                "dialogue_id": f"dlg-{index}",
                "turns": [
                    {"speaker": "USER", "utterance": utterance, "frames": [frame(service, intent, slot)]}
                ],
            }
        )
    return rows


CONFIG = {
    "excludedServices": [],
    "classConstruction": {
        "hiddenIntentSalt": "hidden",
        "unsupportedTargetSalt": "unsupported",
        "minimumSourceIntentRecordCountForHiding": 1,
        "minimumTypedIntentCountPerService": 3,
        "minimumDeclaredIntentCountPerService": 2,
    },
}


class V93OpenSetSourceTests(unittest.TestCase):
    def test_tokenization_handles_camel_case_and_stop_words(self) -> None:
        self.assertEqual(normalized_tokens("Please AddItem to the list"), frozenset({"add", "item", "list"}))

    def test_git_blob_identity(self) -> None:
        data = b"hello\n"
        self.assertEqual(git_blob_sha1(data), hashlib.sha1(b"blob 6\0hello\n").hexdigest())  # noqa: S324

    def test_schema_is_typed_and_unique(self) -> None:
        compiled = compile_schema(SCHEMA)
        self.assertEqual(len(compiled["Alpha_1"]["intent_names"]), 4)
        bad = [dict(SCHEMA[0], intents=[SCHEMA[0]["intents"][0], SCHEMA[0]["intents"][0]])]
        with self.assertRaisesRegex(ValueError, "intent names"):
            compile_schema(bad)

    def test_inventory_contains_all_five_classes_without_language(self) -> None:
        inventory = build_open_set_inventory(SCHEMA, dialogues(), CONFIG)
        self.assertEqual(inventory["eligible_service_count"], 2)
        self.assertEqual(set(inventory["class_counts"]), {
            "known_familiar", "known_unfamiliar", "novel_valid", "unsupported", "insufficient_evidence"
        })
        serialized = str(inventory)
        self.assertNotIn("discard the object", serialized)
        self.assertNotIn("private", serialized)
        self.assertFalse(inventory["contains_language_tokens_slot_values_or_histories"])

    def test_hidden_split_and_unsupported_mapping_are_deterministic(self) -> None:
        first = build_open_set_inventory(SCHEMA, dialogues(), CONFIG)
        second = build_open_set_inventory(SCHEMA, dialogues(), CONFIG)
        self.assertEqual(first["service_splits"], second["service_splits"])
        self.assertEqual(first["candidate_index_sha256"], second["candidate_index_sha256"])


if __name__ == "__main__":
    unittest.main()
