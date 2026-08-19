from __future__ import annotations

import unittest

from v162_fresh_massive_transfer_language_extraction import (
    build_selected_language_artifacts,
    evaluate_extraction_gates,
)


def fixture() -> tuple[dict, dict, list, dict]:
    classes = ["known_familiar", "known_unfamiliar", "novel_valid", "unsupported"]
    source_records = []
    candidates = []
    selected = []
    cursor = 0
    for role, source_partition in (
        ("development_transfer", "dev"),
        ("protected_transfer", "test"),
    ):
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
            canonical_partition = "validation" if source_partition == "dev" else "test"
            overlap = 1 if label == "known_familiar" else 0
            source_records.append(
                {
                    "id": identifier,
                    "locale": "en-US",
                    "partition": source_partition,
                    "scenario": scenario,
                    "intent": intent,
                    "utt": utterance,
                    "annot_utt": "use [device : lamp]",
                }
            )
            candidates.append(
                {
                    "candidate_id": f"massive::{identifier}",
                    "source_id": identifier,
                    "partition": canonical_partition,
                    "scenario": scenario,
                    "intent": intent,
                    "class_label": label,
                    "current_utterance_intent_overlap_count": overlap,
                    "slot_type_count": 1,
                }
            )
            selected.append(
                {
                    "population_id": f"v161::{role}::massive::{identifier}",
                    "candidate_id": f"massive::{identifier}",
                    "source_id": identifier,
                    "role": role,
                    "source_partition": canonical_partition,
                    "class_label": label,
                    "scenario": scenario,
                    "intent": intent,
                    "current_utterance_intent_overlap_count": overlap,
                    "slot_type_count": 1,
                }
            )
            cursor += 1
    source_records.append(
        {
            "id": "unselected",
            "locale": "en-US",
            "partition": "dev",
            "scenario": "catalog",
            "intent": "turn_on",
            "utt": "unselected language",
            "annot_utt": "unselected language",
        }
    )
    population = {"selected_population": selected}
    inventory = {
        "candidate_index": candidates,
        "declared_intents": ["catalog::turn_on", "catalog::turn_off"],
        "hidden_intents": ["catalog::dim_light"],
        "unsupported_scenarios": ["outside"],
    }
    config = {
        "roles": {"development_transfer": {}, "protected_transfer": {}},
        "canonicalSourcePartitionMap": {"dev": "validation", "test": "test"},
        "requiredClasses": classes,
        "extractionGates": {
            "requiredTotalRecordCount": 8,
            "requiredRecordCountPerRole": 4,
            "requiredRecordCountPerClassPerRole": 1,
            "maximumUnselectedLanguageRecordCount": 0,
        },
    }
    return population, inventory, source_records, config


class V162ExtractionTests(unittest.TestCase):
    def test_exact_selected_language_extraction(self) -> None:
        population, inventory, records, config = fixture()
        artifacts = build_selected_language_artifacts(
            population, inventory, records, config
        )
        self.assertEqual(artifacts["total_record_count"], 8)
        self.assertTrue(all(evaluate_extraction_gates(artifacts, config).values()))
        self.assertEqual(artifacts["unselected_language_record_count"], 0)
        self.assertFalse(
            any(
                row["source_id"] == "unselected"
                for rows in artifacts["role_records"].values()
                for row in rows
            )
        )

    def test_structural_mismatch_fails_gate(self) -> None:
        population, inventory, records, config = fixture()
        records[0]["scenario"] = "changed"
        artifacts = build_selected_language_artifacts(
            population, inventory, records, config
        )
        self.assertFalse(artifacts["exact_structural_ground_truth_match"])

    def test_duplicate_selected_identifier_fails_exact_set_gate(self) -> None:
        population, inventory, records, config = fixture()
        population["selected_population"][-1] = dict(
            population["selected_population"][0], role="protected_transfer"
        )
        artifacts = build_selected_language_artifacts(
            population, inventory, records, config
        )
        self.assertFalse(artifacts["exact_selected_identifier_set"])
        self.assertFalse(artifacts["development_protected_role_disjoint"])

    def test_unknown_role_is_rejected(self) -> None:
        population, inventory, records, config = fixture()
        population["selected_population"][0]["role"] = "other"
        with self.assertRaises(ValueError):
            build_selected_language_artifacts(population, inventory, records, config)


if __name__ == "__main__":
    unittest.main()
