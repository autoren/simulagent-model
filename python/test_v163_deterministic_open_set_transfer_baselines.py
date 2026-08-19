from __future__ import annotations

import unittest

from v106_open_world_benchmark import prediction
from v163_deterministic_open_set_transfer_baselines import (
    adapt_development_records,
    deterministic_consensus_prediction,
    evaluate_residual_qualification,
    select_controlled_missing_identifiers,
)


class V163DeterministicTransferTests(unittest.TestCase):
    def test_role_adaptation_is_exact(self) -> None:
        rows = [{"record_id": "a", "role": "development_transfer"}]
        self.assertEqual(adapt_development_records(rows)[0]["role"], "development")
        self.assertEqual(rows[0]["role"], "development_transfer")

    def test_unexpected_role_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            adapt_development_records([{"record_id": "a", "role": "protected_transfer"}])

    def test_consensus_accepts_only_identical_complete_decisions(self) -> None:
        known = prediction("KNOWN", 0.9, known_intent="iot::iot_wemo_on")
        same = prediction("KNOWN", 0.8, known_intent="iot::iot_wemo_on")
        other = prediction("KNOWN", 0.95, known_intent="iot::iot_wemo_off")
        accepted = deterministic_consensus_prediction(known, same)
        self.assertEqual(accepted["status"], "KNOWN")
        self.assertEqual(accepted["confidence"], 0.8)
        self.assertEqual(
            deterministic_consensus_prediction(known, other)["status"], "ABSTAIN"
        )

    def test_control_selection_is_identifier_only_and_deterministic(self) -> None:
        config = {
            "controlledMissingObservation": {
                "count": 2,
                "selectionSalt": "test",
            }
        }
        rows = [
            {"record_id": "a", "utterance": "first"},
            {"record_id": "b", "utterance": "second"},
            {"record_id": "c", "utterance": "third"},
        ]
        first = select_controlled_missing_identifiers(rows, config)
        rows[0]["utterance"] = "changed"
        self.assertEqual(first, select_controlled_missing_identifiers(rows, config))
        self.assertTrue(all(set(row) == {"record_id", "observation_available"} for row in first))

    def test_residual_gates_are_noncompensatory(self) -> None:
        config = {
            "residualQualificationGates": {
                "minimumResidualRecordCount": 2,
                "maximumResidualRecordCount": 5,
                "minimumResidualClassCoverage": 2,
                "minimumNonResidualRecordCount": 2,
                "minimumNonResidualExactDecisionAccuracy": 0.95,
                "maximumNonResidualFalseKnownAcceptanceRate": 0.0,
                "maximumNonResidualMeanRegret": 0.25,
                "maximumConsensusOverallFalseKnownAcceptanceRate": 0.05,
                "maximumConsensusRegretAboveAskAlways": 0.0,
            }
        }
        checks = evaluate_residual_qualification(
            {"record_count": 3, "class_coverage": 2},
            {
                "record_count": 2,
                "exact_decision_accuracy": 1.0,
                "false_known_acceptance_rate": 0.0,
                "mean_regret": 0.0,
            },
            {"false_known_acceptance_rate": 0.0, "mean_regret": 1.0},
            {"mean_regret": 1.0},
            config,
        )
        self.assertTrue(all(checks.values()))


if __name__ == "__main__":
    unittest.main()
