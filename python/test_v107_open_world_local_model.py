from __future__ import annotations

import unittest

from v107_open_world_local_model import aggregate_model_fixtures, evaluate_model_gates


CONFIG = {
    "decisionCosts": {
        "known": {"exact_known": 0.0, "wrong_known": 10.0, "novel": 3.0, "unsupported": 6.0, "abstain": 1.0},
        "novel": {"known": 10.0, "exact_novel": 0.0, "wrong_novel_scenario": 5.0, "unsupported": 4.0, "abstain": 1.5},
        "unsupported": {"known": 10.0, "novel": 4.0, "unsupported": 0.0, "abstain": 1.0},
        "insufficient": {"known": 10.0, "novel": 6.0, "unsupported": 4.0, "abstain": 0.0},
    },
    "developmentGates": {
        "minimumStructuredResponseValidity": 1.0, "minimumObservedStatusMacroF1": 1.0,
        "minimumObservedExactDecisionAccuracy": 1.0, "minimumKnownExactIntentAccuracy": 1.0,
        "minimumNovelStatusRecall": 1.0, "minimumNovelStatusPrecision": 1.0,
        "minimumNovelExactScenarioAccuracy": 1.0, "minimumUnsupportedStatusRecall": 1.0,
        "minimumUnsupportedStatusPrecision": 1.0, "maximumFalseKnownAcceptanceRate": 0.0,
        "maximumConfidenceECE": 0.0, "maximumTopConfidence80PercentError": 0.0,
        "maximumMeanDecisionRegret": 0.0, "minimumControlledMissingObservationAbstentionAccuracy": 1.0,
        "maximumRegretAboveBestNonOracleDeterministicBaseline": 0.0,
    },
    "accessGates": {
        "requiredFixtureCount": 4, "maximumDevelopmentLanguageReadCount": 1,
        "maximumProtectedTestLanguageReadCount": 0, "maximumManualUtteranceInspectionCount": 0,
        "maximumModelLoadCount": 1, "maximumModelGenerationCount": 4,
        "maximumLLMAPICallCount": 0, "maximumAdapterTrainingRunCount": 0,
        "maximumRealServiceCallCount": 0, "maximumExternalSideEffectCount": 0,
    },
}


class V107LocalModelTests(unittest.TestCase):
    def test_perfect_fixtures_pass(self) -> None:
        records = [
            {"record_id": "k", "class_label": "known_familiar", "scenario": "calendar", "intent": "calendar_set"},
            {"record_id": "n", "class_label": "novel_valid", "scenario": "calendar", "intent": "calendar_query"},
            {"record_id": "u", "class_label": "unsupported", "scenario": "email", "intent": "send"},
        ]
        responses = {
            "k": {"status": "KNOWN", "known_intent": "calendar::calendar_set", "novel_scenario": None, "confidence": 1.0},
            "n": {"status": "NOVEL", "known_intent": None, "novel_scenario": "calendar", "confidence": 1.0},
            "u": {"status": "UNSUPPORTED", "known_intent": None, "novel_scenario": None, "confidence": 1.0},
            "m": {"status": "ABSTAIN", "known_intent": None, "novel_scenario": None, "confidence": 1.0},
        }
        fixtures = {key: {"parsed_response": value, "response_valid": True} for key, value in responses.items()}
        metrics = aggregate_model_fixtures(fixtures, records, {"m"}, CONFIG)
        self.assertEqual(metrics["observed_exact_decision_accuracy"], 1.0)
        access = {
            "development_language_read_count": 1, "protected_test_language_read_count": 0,
            "manual_utterance_inspection_count": 0, "model_load_count": 1,
            "model_generation_count": 4, "LLM_API_call_count": 0,
            "adapter_training_run_count": 0, "real_service_call_count": 0,
            "external_side_effect_count": 0,
        }
        self.assertTrue(all(evaluate_model_gates(metrics, 0.0, access, CONFIG).values()))


if __name__ == "__main__":
    unittest.main()
