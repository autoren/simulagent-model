from __future__ import annotations

from io import BytesIO
import json
import tarfile
import unittest

from v100_massive_source import (
    build_massive_source_inventory,
    evaluate_massive_source_gates,
    parse_massive_archive,
    slot_types,
)


def records() -> list[dict]:
    rows = []
    cursor = 0
    for scenario_index in range(5):
        scenario = f"scenario{scenario_index}"
        for intent_suffix, familiar, unfamiliar in (
            ("add_item", "add item now", "include object"),
            ("remove_item", "remove item now", "discard object"),
        ):
            intent = f"{scenario}_{intent_suffix}"
            for partition in ("train", "dev", "test"):
                for utterance in (familiar, unfamiliar):
                    rows.append({
                        "id": str(cursor), "locale": "en-US", "partition": partition,
                        "scenario": scenario, "intent": intent, "utt": utterance,
                        "annot_utt": f"[{scenario}_slot : private value]",
                    })
                    cursor += 1
    return rows


CONFIG = {
    "locale": "en-US",
    "requiredRecordFields": ["id", "locale", "partition", "scenario", "intent", "utt", "annot_utt"],
    "allowedSourcePartitions": ["train", "dev", "test"],
    "canonicalPartitionMap": {"train": "train", "dev": "validation", "test": "test"},
    "servicePartition": {
        "eligibleScenarioMinimumRecordCount": 12,
        "eligibleIntentMinimumRecordCount": 6,
        "unsupportedScenarioSalt": "unsupported",
        "unsupportedScenarioCount": 1,
        "catalogScenarioSalt": "catalog",
        "catalogScenarioCount": 3,
        "hiddenScenarioSalt": "hidden-scenario",
        "hiddenIntentSalt": "hidden-intent",
        "hiddenScenarioCount": 2,
        "hiddenIntentCountPerSelectedScenario": 1,
        "minimumDeclaredIntentCount": 3,
    },
}


class V100MassiveSourceTests(unittest.TestCase):
    def test_slot_markup_parser(self) -> None:
        self.assertEqual(slot_types("set [alarm_name : work] for [time : seven]"), {"alarm_name", "time"})

    def test_archive_requires_one_locale_member(self) -> None:
        buffer = BytesIO()
        payload = (json.dumps(records()[0]) + "\n").encode()
        with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
            info = tarfile.TarInfo("massive/1.1/en-US.jsonl")
            info.size = len(payload)
            archive.addfile(info, BytesIO(payload))
        parsed, member = parse_massive_archive(buffer.getvalue(), "en-US.jsonl")
        self.assertEqual(len(parsed), 1)
        self.assertTrue(member.endswith("en-US.jsonl"))

    def test_structural_partition_produces_four_classes(self) -> None:
        result = build_massive_source_inventory(records(), CONFIG)
        self.assertEqual(result["eligible_scenario_count"], 5)
        self.assertEqual(result["catalog_scenario_count"], 3)
        self.assertEqual(result["unsupported_scenario_count"], 1)
        self.assertEqual(result["hidden_intent_count"], 2)
        self.assertGreaterEqual(result["declared_intent_count"], 3)
        self.assertEqual(set(result["class_counts"]), {
            "known_familiar", "known_unfamiliar", "novel_valid", "unsupported",
        })

    def test_inventory_is_deterministic_and_language_free(self) -> None:
        first = build_massive_source_inventory(records(), CONFIG)
        second = build_massive_source_inventory(records(), CONFIG)
        self.assertEqual(first["candidate_index_sha256"], second["candidate_index_sha256"])
        serialized = str(first)
        self.assertNotIn("include object", serialized)
        self.assertNotIn("private value", serialized)
        self.assertFalse(first["contains_raw_or_annotated_utterances_tokens_or_slot_values"])

    def test_partition_gates_are_noncompensatory(self) -> None:
        config = {
            **CONFIG,
            "sourceGates": {
                "minimumScenarioCount": 5, "minimumIntentCount": 10,
                "minimumSlotTypeCount": 5, "minimumEligibleScenarioCount": 4,
                "requiredCatalogScenarioCount": 3, "requiredUnsupportedScenarioCount": 1,
                "requiredHiddenIntentCount": 2, "minimumDeclaredIntentCount": 3,
                "minimumClassCandidateCount": 4,
                "minimumValidationCandidateCountPerClass": 2,
                "minimumTestCandidateCountPerClass": 2,
                "minimumKnownClassScenarioCoverage": 2,
                "requiredNovelScenarioCoverage": 2,
                "requiredUnsupportedScenarioCoverage": 1,
            },
        }
        inventory = build_massive_source_inventory(records(), config)
        checks = evaluate_massive_source_gates(inventory, config)
        self.assertTrue(all(checks.values()))
        damaged = json.loads(json.dumps(inventory))
        damaged["class_partition_counts"]["novel_valid"]["test"] = 1
        checks = evaluate_massive_source_gates(damaged, config)
        self.assertFalse(checks["novel_valid_test_count"])
        self.assertFalse(all(checks.values()))


if __name__ == "__main__":
    unittest.main()
