from __future__ import annotations

import json
import unittest

from v22r2_grounding import PROJECT_ROOT
from v183_sgd_contract_identifiability_population import (
    _frame_signature,
    _parse_candidate_id,
)


class V183ContractIdentifiabilityPopulationTest(unittest.TestCase):
    def test_candidate_identifier_parser(self) -> None:
        self.assertEqual(
            _parse_candidate_id("sgd::dev::1_00000::000::Weather_1::GetWeather"),
            ("dev", "1_00000", 0, "Weather_1", "GetWeather"),
        )

    def test_frame_signature_excludes_values_service_and_spans(self) -> None:
        frame = {
            "service": "Hidden_9",
            "actions": [
                {"act": "INFORM_INTENT", "slot": "intent", "values": ["Secret"]},
                {"act": "INFORM", "slot": "city", "values": ["Toronto"]},
            ],
            "state": {
                "active_intent": "FindThing",
                "requested_slots": ["price"],
                "slot_values": {"city": ["Toronto"]},
            },
            "slots": [{"slot": "date", "start": 0, "exclusive_end": 5}],
        }
        signature = _frame_signature(frame)
        self.assertEqual(signature["normalized_active_intent_name"], "findthing")
        self.assertEqual(signature["all_observed_slot_names"], ["city", "date", "price"])
        serialized = json.dumps(signature)
        for forbidden in ("Hidden_9", "Secret", "Toronto", "start", "exclusive_end"):
            self.assertNotIn(forbidden, serialized)

    def test_config_declares_closed_authority_boundary(self) -> None:
        config = json.loads(
            (
                PROJECT_ROOT
                / "configs/v183-sgd-contract-identifiability-population.json"
            ).read_text()
        )
        self.assertFalse(config["authorityBoundary"]["languageExtractionAllowedDuringV183"])
        self.assertFalse(config["authorityBoundary"]["modelOrAPIAllowedDuringV183"])
        self.assertFalse(config["authorityBoundary"]["actionOrExecutionAllowed"])
        self.assertFalse(config["decisionRule"]["passAuthorizesImmediateLanguageExtraction"])


if __name__ == "__main__":
    unittest.main()
