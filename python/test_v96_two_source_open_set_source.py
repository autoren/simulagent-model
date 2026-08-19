from __future__ import annotations

import unittest

from v96_two_source_open_set_source import build_two_source_open_set_inventory


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
        for index in range(6)
    ]


def dialogues(service_indexes: range) -> list[dict]:
    result = []
    cursor = 0
    for service_index in service_indexes:
        service = f"Service_{service_index}"
        for intent, utterance in (
            (f"AddThing{service_index}", "add the thing"),
            (f"AddThing{service_index}", "include the object"),
            (f"RemoveThing{service_index}", "remove the thing"),
            (f"RemoveThing{service_index}", "discard the object"),
            ("NONE", "not sure"),
        ):
            result.append({
                "dialogue_id": f"dialogue-{service_index}-{cursor}",
                "turns": [{
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
                }],
            })
            cursor += 1
    return result


CONFIG = {
    "previouslyExposedServices": [],
    "catalogPartition": {
        "eligibleServiceMinimumActivationCount": 4,
        "eligiblePairMinimumActivationCount": 2,
        "catalogServiceSalt": "catalog",
        "catalogServiceCount": 3,
        "hiddenServiceSalt": "hidden-service",
        "hiddenIntentPairSalt": "hidden-pair",
        "hiddenServiceCount": 2,
        "hiddenPairCountPerSelectedService": 1,
        "minimumDeclaredPairCount": 3,
    },
    "unsupportedPartition": {
        "eligibleServiceMinimumActivationCount": 4,
        "eligiblePairMinimumActivationCount": 2,
        "unsupportedServiceSalt": "unsupported",
        "unsupportedServiceCount": 1,
    },
}


class V96TwoSourceOpenSetSourceTests(unittest.TestCase):
    def test_disjoint_sources_produce_all_five_classes(self) -> None:
        result = build_two_source_open_set_inventory(
            schema(), dialogues(range(4)), dialogues(range(4, 6)), CONFIG
        )
        self.assertEqual(result["catalog_service_count"], 3)
        self.assertEqual(result["unsupported_service_count"], 1)
        self.assertEqual(result["hidden_pair_count"], 2)
        self.assertEqual(len(result["hidden_services"]), 2)
        self.assertGreaterEqual(result["declared_supported_pair_count"], 3)
        self.assertEqual(set(result["class_counts"]), {
            "known_familiar", "known_unfamiliar", "novel_valid", "unsupported",
            "insufficient_evidence",
        })
        self.assertFalse(set(result["catalog_services"]) & set(result["unsupported_services"]))

    def test_partition_is_deterministic_and_text_free(self) -> None:
        first = build_two_source_open_set_inventory(
            schema(), dialogues(range(4)), dialogues(range(4, 6)), CONFIG
        )
        second = build_two_source_open_set_inventory(
            schema(), dialogues(range(4)), dialogues(range(4, 6)), CONFIG
        )
        self.assertEqual(first["candidate_index_sha256"], second["candidate_index_sha256"])
        self.assertEqual(first["hidden_pairs"], second["hidden_pairs"])
        serialized = str(first)
        self.assertNotIn("include the object", serialized)
        self.assertNotIn("private", serialized)
        self.assertFalse(first["contains_language_tokens_slot_values_or_histories"])

    def test_non_none_cases_are_activations_and_roles_are_fixed(self) -> None:
        result = build_two_source_open_set_inventory(
            schema(), dialogues(range(4)), dialogues(range(4, 6)), CONFIG
        )
        for row in result["candidate_index"]:
            if row["class_label"] != "insufficient_evidence":
                self.assertTrue(row["source_intent_activation"])
            if row["class_label"] == "unsupported":
                self.assertEqual(row["source_role"], "unsupported")
            else:
                self.assertEqual(row["source_role"], "catalog")


if __name__ == "__main__":
    unittest.main()
