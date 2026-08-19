from __future__ import annotations

import json
import unittest

from v115_contrastive_catalog_fit import (
    classify_v115, contrastive_policy_prediction, render_contrastive_prompt, reviewed_choice,
    validate_and_expand_contrastive,
)


CHOICES = {"choices": [
    {"choice_id": "K00", "kind": "KNOWN", "intent_id": "calendar::set", "scenario": "calendar", "intent_label": "set", "slot_types": []},
    {"choice_id": "K01", "kind": "KNOWN", "intent_id": "calendar::remove", "scenario": "calendar", "intent_label": "remove", "slot_types": []},
    {"choice_id": "N00", "kind": "NOVEL", "scenario": "calendar", "meaning": "new"},
    {"choice_id": "U00", "kind": "UNSUPPORTED", "meaning": "outside"},
    {"choice_id": "A00", "kind": "ABSTAIN", "meaning": "insufficient"},
]}
CONFIG = {
    "contrastiveInterface": {
        "outputKeys": ["verdict_id", "selected_choice_id", "verdict_confidence", "novel_probability"],
        "verdicts": {key: key for key in ("C00", "O00", "N00", "U00", "A00")},
    },
    "prompt": {"missingObservationSentinel": "[NONE]"},
}


class V115ContrastiveTests(unittest.TestCase):
    def test_review_uses_first_pass_known_then_retrieval(self) -> None:
        first = {"status": "KNOWN", "known_intent": "calendar::remove"}
        self.assertEqual(reviewed_choice(first, "calendar::set", CHOICES, True)["choice_id"], "K01")
        abstain = {"status": "ABSTAIN", "known_intent": None}
        self.assertEqual(reviewed_choice(abstain, "calendar::set", CHOICES, True)["choice_id"], "K00")

    def test_prompt_hides_missing_observation(self) -> None:
        candidate = CHOICES["choices"][0]
        payload = json.loads(render_contrastive_prompt(CHOICES, candidate, None, False, CONFIG))
        self.assertEqual(payload["user_utterance"], "[NONE]")
        with self.assertRaises(ValueError):
            render_contrastive_prompt(CHOICES, candidate, "secret", False, CONFIG)

    def test_every_consistent_verdict_expands(self) -> None:
        candidate = CHOICES["choices"][0]
        cases = {
            "C00": ("K00", "KNOWN"), "O00": ("K01", "KNOWN"),
            "N00": ("N00", "NOVEL"), "U00": ("U00", "UNSUPPORTED"),
            "A00": ("A00", "ABSTAIN"),
        }
        for verdict, (selected, status) in cases.items():
            parsed, evidence, valid, reason = validate_and_expand_contrastive({
                "verdict_id": verdict, "selected_choice_id": selected,
                "verdict_confidence": 0.8, "novel_probability": 0.7 if verdict == "N00" else 0.1,
            }, candidate, CHOICES, CONFIG)
            self.assertTrue(valid, reason)
            self.assertEqual(parsed["status"], status)
            self.assertEqual(evidence["novel_candidate"], verdict == "N00")

    def test_inconsistent_or_alias_output_fails_closed(self) -> None:
        candidate = CHOICES["choices"][0]
        parsed, evidence, valid, _ = validate_and_expand_contrastive({
            "verdict_id": "C00", "selected_choice_id": "K01",
            "verdict_confidence": 0.9, "novel_probability": 0.1,
        }, candidate, CHOICES, CONFIG)
        self.assertFalse(valid)
        self.assertEqual(parsed["status"], "ABSTAIN")
        self.assertEqual(evidence["novel_evidence_probability"], 0.0)

    def test_policy_requires_two_pass_confirmation(self) -> None:
        known = {"status": "KNOWN", "known_intent": "calendar::set", "novel_scenario": None, "confidence": 0.9}
        review = {"status": "KNOWN", "known_intent": "calendar::set", "novel_scenario": None, "confidence": 0.8}
        evidence = {"verdict_id": "C00", "novel_candidate": False, "novel_evidence_probability": 0.1}
        action, _ = contrastive_policy_prediction(known, review, evidence)
        self.assertEqual(action["status"], "KNOWN")
        action, _ = contrastive_policy_prediction(known, review, {**evidence, "verdict_id": "O00"})
        self.assertEqual(action["status"], "ABSTAIN")

    def test_induction_is_never_authorized_by_v115(self) -> None:
        summary = {
            "contrastive_evidence_gates": {"evidence": True},
            "combined_quality_gates": {"policy": True},
            "base_quality_gates": {"base": False},
            "access_gates": {"access": True},
        }
        result = classify_v115(summary)
        self.assertTrue(result["seek_independent_source_transfer"])
        self.assertFalse(result["schema_induction_authorized"])


if __name__ == "__main__":
    unittest.main()
