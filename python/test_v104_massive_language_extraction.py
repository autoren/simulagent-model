from __future__ import annotations

import unittest

from v104_massive_language_extraction import (
    build_selected_language_artifacts,
    evaluate_extraction_gates,
    parse_annotated_slots,
)


def fixture() -> tuple[dict, dict, list, dict]:
    classes = ["known_familiar", "known_unfamiliar", "novel_valid", "unsupported"]
    records = []
    candidates = []
    selected = []
    cursor = 0
    for role, partition in (("development", "dev"), ("protected_test", "test")):
        for label in classes:
            identifier = str(cursor)
            if label == "known_familiar":
                scenario, intent, utterance = "catalog", "turn_on", "turn on lamp"
            elif label == "known_unfamiliar":
                scenario, intent, utterance = "catalog", "turn_off", "disable lamp"
            elif label == "novel_valid":
                scenario, intent, utterance = "catalog", "dim_light", "lower the lamp"
            else:
                scenario, intent, utterance = "outside", "send_mail", "send mail"
            overlap = 1 if label == "known_familiar" else 0
            records.append({
                "id": identifier, "locale": "en-US", "partition": partition,
                "scenario": scenario, "intent": intent, "utt": utterance,
                "annot_utt": "use [device : lamp]",
            })
            candidates.append({
                "candidate_id": f"massive::{identifier}", "source_id": identifier,
                "partition": "validation" if partition == "dev" else "test",
                "scenario": scenario, "intent": intent,
                "class_label": label, "current_utterance_intent_overlap_count": overlap,
                "slot_type_count": 1,
            })
            selected.append({
                "population_id": f"v101::{role}::massive::{identifier}",
                "candidate_id": f"massive::{identifier}", "source_id": identifier,
                "role": role,
                "source_partition": "validation" if partition == "dev" else "test",
                "class_label": label,
                "scenario": scenario, "intent": intent,
                "current_utterance_intent_overlap_count": overlap, "slot_type_count": 1,
            })
            cursor += 1
    population = {"selected_population": selected}
    inventory = {
        "candidate_index": candidates,
        "declared_intents": ["catalog::turn_on", "catalog::turn_off"],
        "hidden_intents": ["catalog::dim_light"],
        "unsupported_scenarios": ["outside"],
    }
    config = {
        "roles": {"development": {}, "protected_test": {}},
        "canonicalSourcePartitionMap": {"dev": "validation", "test": "test"},
        "requiredClasses": classes,
        "extractionGates": {
            "requiredTotalRecordCount": 8, "requiredRecordCountPerRole": 4,
            "requiredRecordCountPerClassPerRole": 1, "maximumUnselectedLanguageRecordCount": 0,
        },
    }
    return population, inventory, records, config


class V104ExtractionTests(unittest.TestCase):
    def test_slot_parser(self) -> None:
        self.assertEqual(parse_annotated_slots("use [device : lamp] at [time : seven]"), [
            {"slot_type": "device", "slot_value": "lamp"},
            {"slot_type": "time", "slot_value": "seven"},
        ])

    def test_exact_selected_language_extraction(self) -> None:
        population, inventory, records, config = fixture()
        artifacts = build_selected_language_artifacts(population, inventory, records, config)
        self.assertEqual(artifacts["total_record_count"], 8)
        self.assertTrue(all(evaluate_extraction_gates(artifacts, config).values()))
        self.assertEqual(artifacts["unselected_language_record_count"], 0)

    def test_structural_mismatch_fails_gate(self) -> None:
        population, inventory, records, config = fixture()
        records[0]["scenario"] = "changed"
        artifacts = build_selected_language_artifacts(population, inventory, records, config)
        self.assertFalse(artifacts["exact_structural_ground_truth_match"])


if __name__ == "__main__":
    unittest.main()
