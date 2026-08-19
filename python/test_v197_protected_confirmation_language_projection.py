from __future__ import annotations

import unittest

from v197_protected_confirmation_language_projection import audit_projection, build_projection


class V197ProjectionTest(unittest.TestCase):
    def test_projection_is_selected_by_id_and_sanitized(self) -> None:
        source = {"records": [
            {"record_id": "a", "observation_available": True, "conversation": [{"speaker": "USER", "utterance": "x"}], "presented_candidate_choice_id": "K1"},
            {"record_id": "b", "observation_available": False, "conversation": None, "presented_candidate_choice_id": "K2"},
            {"record_id": "c", "observation_available": True, "conversation": [{"speaker": "USER", "utterance": "z"}], "presented_candidate_choice_id": "K3"},
        ]}
        identities = {"records": [
            {"record_id": "a", "observation_available": True},
            {"record_id": "b", "observation_available": False},
        ]}
        config = {
            "projection": {"outputRole": "confirmation"},
            "projectionGates": {
                "requiredInputRecordCount": 3, "requiredSelectedRecordCount": 2,
                "requiredSelectedObservedCount": 1, "requiredSelectedMissingCount": 1,
                "requiredUnselectedReadButNotEmittedCount": 1,
                "requiredIdentifierReconstructionRate": 1.0, "requiredConversationProjectionExactness": 1.0,
                "requiredMissingConversationNullRate": 1.0, "maximumForbiddenFieldOccurrenceCount": 0,
                "maximumUnselectedLanguageEmissionCount": 0, "maximumManualLanguageInspectionCount": 0,
                "maximumPolicyScoreCount": 0, "maximumModelLoadCount": 0, "maximumModelGenerationCount": 0,
                "maximumAPICallCount": 0, "maximumTrainingRunCount": 0, "maximumOntologyRegistrationCount": 0,
                "maximumTrustedStateMutationCount": 0, "maximumRealServiceCallCount": 0,
                "maximumExternalSideEffectCount": 0, "maximumActualExecutionCount": 0,
            },
        }
        projected = build_projection(source, identities, config)
        self.assertTrue(audit_projection(projected, config)["passed"])
        self.assertEqual([row["record_id"] for row in projected["language"]["records"]], ["a", "b"])
        self.assertNotIn("presented_candidate_choice_id", projected["language"]["records"][0])


if __name__ == "__main__":
    unittest.main()
