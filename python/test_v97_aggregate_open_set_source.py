from __future__ import annotations

import unittest

from v97_aggregate_open_set_source import build_aggregate_open_set_inventory


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


def shard(shard_index: int, service_indexes: range) -> tuple[str, list[dict]]:
    payload = []
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
            payload.append({
                "dialogue_id": f"dialogue-{shard_index}-{service_index}-{cursor}",
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
    return f"dev/dialogues_{shard_index:03d}.json", payload


CONFIG = {
    "previouslyExposedServices": [],
    "servicePartition": {
        "eligibleServiceMinimumActivationCount": 4,
        "eligiblePairMinimumActivationCount": 2,
        "unsupportedServiceSalt": "unsupported",
        "unsupportedServiceCount": 1,
        "catalogServiceSalt": "catalog",
        "catalogServiceCount": 3,
        "hiddenServiceSalt": "hidden-service",
        "hiddenIntentPairSalt": "hidden-pair",
        "hiddenServiceCount": 2,
        "hiddenPairCountPerSelectedService": 1,
        "minimumDeclaredPairCount": 3,
    },
}


class V97AggregateOpenSetSourceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.pool = [shard(9, range(3)), shard(10, range(3, 6))]

    def test_aggregate_service_partition_produces_all_classes(self) -> None:
        result = build_aggregate_open_set_inventory(schema(), self.pool, CONFIG)
        self.assertEqual(result["aggregate_shard_count"], 2)
        self.assertEqual(result["eligible_fresh_service_count"], 6)
        self.assertEqual(result["catalog_service_count"], 3)
        self.assertEqual(result["unsupported_service_count"], 1)
        self.assertEqual(result["hidden_pair_count"], 2)
        self.assertGreaterEqual(result["declared_supported_pair_count"], 3)
        self.assertEqual(set(result["class_counts"]), {
            "known_familiar", "known_unfamiliar", "novel_valid", "unsupported",
            "insufficient_evidence",
        })

    def test_partition_is_deterministic_service_disjoint_and_text_free(self) -> None:
        first = build_aggregate_open_set_inventory(schema(), self.pool, CONFIG)
        second = build_aggregate_open_set_inventory(schema(), self.pool, CONFIG)
        self.assertEqual(first["candidate_index_sha256"], second["candidate_index_sha256"])
        self.assertEqual(first["hidden_pairs"], second["hidden_pairs"])
        self.assertFalse(set(first["catalog_services"]) & set(first["unsupported_services"]))
        serialized = str(first)
        self.assertNotIn("include the object", serialized)
        self.assertNotIn("private", serialized)
        self.assertFalse(first["contains_language_tokens_slot_values_or_histories"])

    def test_non_none_cases_are_source_activations(self) -> None:
        result = build_aggregate_open_set_inventory(schema(), self.pool, CONFIG)
        self.assertTrue(all(
            row["source_intent_activation"]
            for row in result["candidate_index"]
            if row["class_label"] != "insufficient_evidence"
        ))


if __name__ == "__main__":
    unittest.main()
