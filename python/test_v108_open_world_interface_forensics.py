from __future__ import annotations

import json
import unittest

from v108_open_world_interface_forensics import classify_response_shape, diagnostic_canonicalize


CATALOG = {
    "scenarios": ["calendar"],
    "intents": [{
        "intent_id": "calendar::calendar_set", "scenario": "calendar", "intent": "calendar_set",
    }],
}
CONFIG = {
    "responseContract": {
        "requiredKeys": ["status", "known_intent", "novel_scenario", "confidence"],
        "allowedStatuses": ["KNOWN", "NOVEL", "UNSUPPORTED", "ABSTAIN"],
        "confidenceMinimum": 0.0, "confidenceMaximum": 1.0,
        "invalidResponseFallback": {"status": "ABSTAIN", "known_intent": None, "novel_scenario": None, "confidence": 0.0},
    }
}


class V108ForensicsTests(unittest.TestCase):
    def test_unique_local_name_is_canonicalized_without_semantic_inference(self) -> None:
        raw = json.dumps({
            "status": "KNOWN", "known_intent": "calendar_set",
            "novel_scenario": None, "confidence": 0.9,
        })
        self.assertEqual(classify_response_shape(raw, CATALOG), "known_local_name_only")
        result = diagnostic_canonicalize(raw, CATALOG, CONFIG)
        self.assertTrue(result["valid"])
        self.assertEqual(result["parsed_response"]["known_intent"], "calendar::calendar_set")
        self.assertEqual(result["parsed_response"]["confidence"], 0.9)

    def test_matching_redundant_scenario_is_removed_only_for_resolved_known(self) -> None:
        raw = json.dumps({
            "status": "KNOWN", "known_intent": "calendar_set",
            "novel_scenario": "calendar", "confidence": 0.8,
        })
        result = diagnostic_canonicalize(raw, CATALOG, CONFIG)
        self.assertTrue(result["valid"])
        self.assertEqual(result["parsed_response"]["novel_scenario"], None)
        self.assertEqual(len(result["transforms"]), 2)

    def test_unknown_identifier_remains_invalid_and_falls_back(self) -> None:
        raw = json.dumps({
            "status": "KNOWN", "known_intent": "invented",
            "novel_scenario": None, "confidence": 0.9,
        })
        result = diagnostic_canonicalize(raw, CATALOG, CONFIG)
        self.assertFalse(result["valid"])
        self.assertEqual(result["parsed_response"]["status"], "ABSTAIN")

    def test_non_string_identifier_remains_invalid_without_crashing(self) -> None:
        raw = json.dumps({
            "status": "KNOWN", "known_intent": ["calendar_set"],
            "novel_scenario": "calendar", "confidence": 0.9,
        })
        result = diagnostic_canonicalize(raw, CATALOG, CONFIG)
        self.assertFalse(result["valid"])
        self.assertEqual(result["parsed_response"]["status"], "ABSTAIN")


if __name__ == "__main__":
    unittest.main()
