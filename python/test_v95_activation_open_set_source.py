from __future__ import annotations

import unittest

from v95_activation_open_set_source import build_activation_open_set_inventory


def schema() -> list[dict]:
    return [
        {
            "service_name": f"Service_{index}",
            "slots": [{"name": "item"}],
            "intents": [
                {"name": f"AddThing{index}", "description": "add a thing"},
                {"name": f"RemoveThing{index}", "description": "remove a thing"},
            ],
        }
        for index in range(5)
    ]


def dialogues() -> list[dict]:
    result = []
    cursor = 0
    for service_index in range(5):
        service = f"Service_{service_index}"
        for intent, utterance in (
            (f"AddThing{service_index}", "add the thing"),
            (f"AddThing{service_index}", "include the object"),
            (f"RemoveThing{service_index}", "remove the thing"),
            (f"RemoveThing{service_index}", "discard the object"),
            ("NONE", "not sure"),
        ):
            turns = [
                {
                    "speaker": "USER",
                    "utterance": utterance,
                    "frames": [{
                        "service": service,
                        "state": {
                            "active_intent": intent,
                            "requested_slots": [],
                            "slot_values": {"item": ["private"]},
                        },
                    }],
                }
            ]
            if intent != "NONE":
                turns.append(
                    {
                        "speaker": "USER",
                        "utterance": "and use another private value",
                        "frames": [{
                            "service": service,
                            "state": {
                                "active_intent": intent,
                                "requested_slots": [],
                                "slot_values": {"item": ["secret"]},
                            },
                        }],
                    }
                )
            result.append({"dialogue_id": f"dialogue-{cursor}", "turns": turns})
            cursor += 1
    return result


CONFIG = {
    "previouslyExposedServices": [],
    "catalogPartition": {
        "eligibleServiceMinimumActivationCount": 4,
        "eligiblePairMinimumActivationCount": 2,
        "unsupportedServiceSalt": "unsupported",
        "unsupportedServiceCount": 1,
        "minimumCatalogServiceCount": 3,
        "hiddenServiceSalt": "hidden-service",
        "hiddenIntentPairSalt": "hidden-pair",
        "hiddenServiceCount": 2,
        "hiddenPairCountPerSelectedService": 1,
        "minimumDeclaredPairCount": 3,
    },
}


class V95ActivationOpenSetSourceTests(unittest.TestCase):
    def test_service_stratified_partition_produces_all_classes(self) -> None:
        result = build_activation_open_set_inventory(schema(), dialogues(), CONFIG)
        self.assertEqual(result["eligible_fresh_service_count"], 5)
        self.assertEqual(result["unsupported_service_count"], 1)
        self.assertEqual(result["catalog_service_count"], 4)
        self.assertEqual(result["hidden_pair_count"], 2)
        self.assertEqual(len(result["hidden_services"]), 2)
        self.assertGreaterEqual(result["declared_supported_pair_count"], 3)
        self.assertEqual(set(result["class_counts"]), {
            "known_familiar", "known_unfamiliar", "novel_valid", "unsupported",
            "insufficient_evidence",
        })

    def test_continuations_do_not_enter_non_none_classes(self) -> None:
        result = build_activation_open_set_inventory(schema(), dialogues(), CONFIG)
        non_none = [
            row for row in result["candidate_index"]
            if row["class_label"] != "insufficient_evidence"
        ]
        self.assertTrue(non_none)
        self.assertTrue(all(row["source_intent_activation"] for row in non_none))
        self.assertEqual(result["source_intent_activation_count"], 20)

    def test_partition_is_deterministic_current_turn_only_and_text_free(self) -> None:
        first = build_activation_open_set_inventory(schema(), dialogues(), CONFIG)
        second = build_activation_open_set_inventory(schema(), dialogues(), CONFIG)
        self.assertEqual(first["candidate_index_sha256"], second["candidate_index_sha256"])
        self.assertEqual(first["hidden_pairs"], second["hidden_pairs"])
        serialized = str(first)
        self.assertNotIn("include the object", serialized)
        self.assertNotIn("private", serialized)
        self.assertTrue(first["lexical_separation_uses_current_turn_only"])
        self.assertFalse(first["contains_language_tokens_slot_values_or_histories"])

    def test_NONE_only_maps_to_insufficient_inside_catalog(self) -> None:
        result = build_activation_open_set_inventory(schema(), dialogues(), CONFIG)
        for row in result["candidate_index"]:
            if row["gold_source_intent"] == "NONE":
                self.assertEqual(row["class_label"], "insufficient_evidence")


if __name__ == "__main__":
    unittest.main()
