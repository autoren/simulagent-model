from __future__ import annotations

import json
import unittest

from v109_open_world_typed_choice import (
    compile_choice_catalog, render_choice_prompt, validate_and_expand_choice,
)


CATALOG = {
    "scenarios": ["calendar"],
    "intents": [{
        "intent_id": "calendar::calendar_set", "scenario": "calendar",
        "intent": "calendar_set", "slot_types": ["date"],
    }],
}
CONFIG = {
    "typedChoiceInterface": {
        "knownChoicePrefix": "K", "novelScenarioChoicePrefix": "N",
        "unsupportedChoiceId": "U00", "insufficientEvidenceChoiceId": "A00",
        "choiceWidth": 2, "requiredChoiceCount": 4,
        "outputKeys": ["choice_id", "confidence"],
    },
    "prompt": {"missingObservationSentinel": "[NONE]"},
}


class V109TypedChoiceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.choices = compile_choice_catalog(CATALOG, CONFIG)

    def test_catalog_is_complete_and_unique(self) -> None:
        self.assertEqual([row["choice_id"] for row in self.choices["choices"]], ["K00", "N00", "U00", "A00"])

    def test_known_choice_expands_to_qualified_intent(self) -> None:
        parsed, valid, reason = validate_and_expand_choice(
            json.dumps({"choice_id": "K00", "confidence": 0.8}), self.choices, CONFIG,
        )
        self.assertTrue(valid)
        self.assertEqual(reason, "valid")
        self.assertEqual(parsed["known_intent"], "calendar::calendar_set")

    def test_alias_is_rejected_and_fails_closed(self) -> None:
        parsed, valid, _ = validate_and_expand_choice(
            json.dumps({"choice_id": "calendar_set", "confidence": 0.8}), self.choices, CONFIG,
        )
        self.assertFalse(valid)
        self.assertEqual(parsed, {"status": "ABSTAIN", "known_intent": None, "novel_scenario": None, "confidence": 0.0})

    def test_novel_unsupported_and_abstain_expand(self) -> None:
        expected = {"N00": "NOVEL", "U00": "UNSUPPORTED", "A00": "ABSTAIN"}
        for choice_id, status in expected.items():
            parsed, valid, _ = validate_and_expand_choice(
                {"choice_id": choice_id, "confidence": 1}, self.choices, CONFIG,
            )
            self.assertTrue(valid)
            self.assertEqual(parsed["status"], status)

    def test_prompt_never_exposes_utterance_when_missing(self) -> None:
        payload = json.loads(render_choice_prompt(self.choices, None, False, CONFIG))
        self.assertEqual(payload["user_utterance"], "[NONE]")
        with self.assertRaises(ValueError):
            render_choice_prompt(self.choices, "secret", False, CONFIG)


if __name__ == "__main__":
    unittest.main()
