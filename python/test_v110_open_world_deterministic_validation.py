from __future__ import annotations

import unittest

from v110_open_world_deterministic_validation import (
    confidence_abstention_prediction, deterministic_novelty_gate_prediction,
    llm_plus_validation_prediction, split_secondary_development,
)


class V110DeterministicValidationTests(unittest.TestCase):
    def test_secondary_split_is_balanced_disjoint_and_deterministic(self) -> None:
        records = [
            {"record_id": f"{label}-{index}", "class_label": label}
            for label in ("known_familiar", "known_unfamiliar", "novel_valid", "unsupported")
            for index in range(4)
        ]
        config = {"secondaryDevelopmentSplit": {
            "classes": ["known_familiar", "known_unfamiliar", "novel_valid", "unsupported"],
            "calibrationCountPerClass": 2, "evaluationCountPerClass": 2, "salt": "test",
        }}
        first = split_secondary_development(records, config)
        second = split_secondary_development(list(reversed(records)), config)
        self.assertEqual(first["membership"], second["membership"])
        self.assertEqual(first["counts"], {"calibration": 8, "evaluation": 8})
        self.assertFalse(
            {row["record_id"] for row in first["calibration"]}
            & {row["record_id"] for row in first["evaluation"]}
        )

    def test_confidence_abstention_fails_closed(self) -> None:
        direct = {"status": "KNOWN", "known_intent": "a::b", "novel_scenario": None, "confidence": 0.7}
        self.assertEqual(confidence_abstention_prediction(direct, 0.8, 0.0)["status"], "ABSTAIN")
        self.assertEqual(confidence_abstention_prediction(direct, 0.7, 0.0), direct)

    def test_novelty_gate_uses_only_retrieval_novel_decision(self) -> None:
        direct = {"status": "KNOWN", "known_intent": "a::b", "novel_scenario": None, "confidence": 0.9}
        novel = {"status": "NOVEL", "known_intent": None, "novel_scenario": "a", "confidence": 0.6}
        unsupported = {"status": "UNSUPPORTED", "known_intent": None, "novel_scenario": None, "confidence": 0.8}
        self.assertEqual(deterministic_novelty_gate_prediction(direct, novel), novel)
        self.assertEqual(deterministic_novelty_gate_prediction(direct, unsupported), direct)

    def test_llm_validation_accepts_novel_or_exact_agreement_and_abstains_otherwise(self) -> None:
        known = {"status": "KNOWN", "known_intent": "a::b", "novel_scenario": None, "confidence": 0.9}
        same = {**known, "confidence": 0.7}
        other = {"status": "KNOWN", "known_intent": "a::c", "novel_scenario": None, "confidence": 0.8}
        novel = {"status": "NOVEL", "known_intent": None, "novel_scenario": "a", "confidence": 0.6}
        self.assertEqual(llm_plus_validation_prediction(known, novel), novel)
        self.assertEqual(llm_plus_validation_prediction(known, same)["confidence"], 0.7)
        self.assertEqual(llm_plus_validation_prediction(known, other)["status"], "ABSTAIN")


if __name__ == "__main__":
    unittest.main()
