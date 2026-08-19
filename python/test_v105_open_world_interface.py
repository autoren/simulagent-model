from __future__ import annotations

import unittest

from v105_open_world_interface import (
    compile_visible_catalog, complete_hypothesis_universe, evaluate_interface_gates, render_prompt,
    select_controlled_insufficient_identifiers, validate_response,
)


CONFIG = {
    "visibleDeclaredIntents": ["calendar::set_event"],
    "visibleScenarios": ["calendar"],
    "hiddenGroundTruthMustNotAppearInVisibleCatalog": ["calendar::query_event", "email"],
    "responseContract": {
        "requiredKeys": ["status", "known_intent", "novel_scenario", "confidence"],
        "allowedStatuses": ["KNOWN", "NOVEL", "UNSUPPORTED", "ABSTAIN"],
        "confidenceMinimum": 0.0, "confidenceMaximum": 1.0,
        "invalidResponseFallback": {
            "status": "ABSTAIN", "known_intent": None,
            "novel_scenario": None, "confidence": 0.0,
        },
    },
    "promptContract": {
        "missingObservationSentinel": "[NO_USER_UTTERANCE_AVAILABLE]",
        "instruction": "classify",
    },
    "controlledInsufficientEvidence": {"recordsPerRole": 1, "selectionSalt": "test"},
    "populationRoles": ["development", "protected_test"],
}


class V105InterfaceTests(unittest.TestCase):
    def test_catalog_is_training_only_typed_and_hidden_free(self) -> None:
        records = [{
            "partition": "train", "scenario": "calendar", "intent": "set_event",
            "annot_utt": "set [date : tomorrow]",
        }]
        inventory = {"declared_intents": ["calendar::set_event"]}
        compiled = compile_visible_catalog(records, inventory, CONFIG)
        self.assertEqual(compiled["catalog"]["intents"][0]["slot_types"], ["date"])
        self.assertEqual(compiled["hidden_or_unsupported_schema_leak_count"], 0)
        self.assertNotIn("query_event", str(compiled))

    def test_hypothesis_universe_and_response_validation(self) -> None:
        catalog = {
            "scenarios": ["calendar"],
            "intents": [{"intent_id": "calendar::set_event"}],
        }
        self.assertEqual(len(complete_hypothesis_universe(catalog)), 4)
        valid, passed, _ = validate_response({
            "status": "KNOWN", "known_intent": "calendar::set_event",
            "novel_scenario": None, "confidence": 0.8,
        }, catalog, CONFIG)
        self.assertTrue(passed)
        self.assertEqual(valid["status"], "KNOWN")
        fallback, passed, reason = validate_response("not-json", catalog, CONFIG)
        self.assertFalse(passed)
        self.assertEqual(reason, "invalid_json")
        self.assertEqual(fallback["status"], "ABSTAIN")

    def test_missing_observation_never_includes_source_utterance(self) -> None:
        catalog = {"scenarios": ["calendar"], "intents": []}
        prompt = render_prompt(catalog, "private source words", False, CONFIG)
        self.assertNotIn("private source words", prompt)
        self.assertIn("NO_USER_UTTERANCE_AVAILABLE", prompt)

    def test_controlled_identifiers_are_hash_selected_and_language_free(self) -> None:
        population = {"selected_population": [
            {"role": "development", "candidate_id": "d"},
            {"role": "protected_test", "candidate_id": "t"},
        ]}
        control = select_controlled_insufficient_identifiers(population, CONFIG)
        self.assertEqual(control["role_counts"], {"development": 1, "protected_test": 1})
        self.assertFalse(control["contains_source_language"])

    def test_interface_gates_cover_structure_and_both_roles(self) -> None:
        compiled = {
            "catalog": {
                "scenarios": ["calendar"],
                "intents": [{"intent_id": "calendar::set_event"}],
                "visible_unique_slot_type_count": 1,
            },
            "hidden_or_unsupported_schema_leak_count": 0,
        }
        control = {
            "contains_source_language": False,
            "role_counts": {"development": 1, "protected_test": 1},
        }
        config = dict(CONFIG)
        config["interfaceGates"] = {
            "requiredVisibleScenarioCount": 1,
            "requiredVisibleIntentCount": 1,
            "minimumVisibleUniqueSlotTypeCount": 1,
            "requiredSafeHypothesisCount": 4,
            "requiredControlledInsufficientRecordsPerRole": 1,
            "maximumHiddenOrUnsupportedSchemaLeakCount": 0,
        }
        self.assertTrue(all(evaluate_interface_gates(compiled, 4, control, config).values()))


if __name__ == "__main__":
    unittest.main()
