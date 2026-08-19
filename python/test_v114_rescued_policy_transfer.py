from __future__ import annotations

import unittest

from v114_rescued_policy_transfer import (
    classify_transfer, merged_excluded_population, rescue_mechanism_gates, select_v114_population,
)


class V114TransferTests(unittest.TestCase):
    def test_excluded_populations_merge_without_overlap(self) -> None:
        merged = merged_excluded_population([
            {"selected_population": [{"candidate_id": "a"}]},
            {"selected_population": [{"candidate_id": "b"}]},
        ])
        self.assertEqual({row["candidate_id"] for row in merged["selected_population"]}, {"a", "b"})

    def test_overlapping_exclusions_fail_closed(self) -> None:
        with self.assertRaises(ValueError):
            merged_excluded_population([
                {"selected_population": [{"candidate_id": "a"}]},
                {"selected_population": [{"candidate_id": "a"}]},
            ])

    def test_v114_selector_relabels_v112_selector_output(self) -> None:
        classes = ["known_familiar"]
        inventory = {"candidate_index": [{
            "candidate_id": "massive::1", "source_id": "1", "partition": "test",
            "class_label": "known_familiar", "scenario": "alarm", "intent": "alarm_set",
            "current_utterance_intent_overlap_count": 1, "slot_type_count": 1,
        }]}
        config = {
            "freshPopulation": {
                "sourcePartition": "test", "baseSalt": "test", "classes": classes,
                "recordCountPerClass": 1, "requiredScenarioCoverage": {"known_familiar": 1},
            },
            "extraction": {"role": "development_transfer_2"},
        }
        selected = select_v114_population(inventory, {"selected_population": []}, config)
        row = selected["selected_population"][0]
        self.assertEqual(row["role"], "development_transfer_2")
        self.assertTrue(row["population_id"].startswith("v114::development_transfer_2::"))

    def test_mechanism_gate_separates_opportunity_from_safety(self) -> None:
        diagnostics = {
            "eligible_disagreement_count": 7, "triggered_rescue_count": 4,
            "rescue_precision": 1.0, "net_corrected_errors": 4,
            "novel_evidence_exactly_unchanged": True,
            "metric_deltas_rescued_minus_baseline": {
                "known_exact_intent_accuracy": 0.01,
                "top_confidence_80_percent_error": -0.01,
                "mean_regret": -0.01, "false_known_acceptance_rate": 0.0,
                "unsupported_precision": 0.0, "unsupported_recall": 0.0,
            },
        }
        config = {"pairedRescueEvaluation": {
            "minimumEligibleDisagreementCountForMechanismConclusion": 8,
            "minimumTriggeredRescueCountForMechanismConclusion": 4,
            "minimumRescuePrecision": 0.75, "minimumNetCorrectedErrors": 1,
            "minimumKnownExactIntentAccuracyDelta": 0.0,
            "maximumTopConfidence80PercentErrorDelta": 0.0,
            "maximumMeanRegretDelta": 0.0,
            "maximumFalseKnownAcceptanceRateDelta": 0.0,
            "minimumUnsupportedPrecisionDelta": 0.0,
            "minimumUnsupportedRecallDelta": 0.0,
            "requireExactNovelEvidenceIdentity": True,
        }}
        gates = rescue_mechanism_gates(diagnostics, config)
        self.assertFalse(gates["minimum_eligible_disagreement_opportunity"])
        self.assertTrue(all(value for key, value in gates.items() if key != "minimum_eligible_disagreement_opportunity"))

    def test_full_policy_can_pass_while_mechanism_is_inconclusive(self) -> None:
        gates = {
            "minimum_eligible_disagreement_opportunity": False,
            "minimum_triggered_rescue_opportunity": False,
            "minimum_rescue_precision": True, "minimum_net_corrected_errors": True,
            "known_accuracy_not_worse": True, "selective_error_not_worse": True,
            "mean_regret_not_worse": True, "false_known_acceptance_not_worse": True,
            "unsupported_precision_not_worse": True, "unsupported_recall_not_worse": True,
            "novel_evidence_exactly_unchanged": True,
        }
        result = classify_transfer(True, True, True, True, {
            "mechanism_gates": gates, "opportunity_sufficient": False, "mechanism_pass": False,
        })
        self.assertEqual(result["mechanism_status"], "inconclusive_insufficient_opportunity")
        self.assertTrue(result["preregister_sandboxed_typed_induction_feasibility"])

    def test_any_paired_harm_rejects_rescue(self) -> None:
        gates = {
            "minimum_eligible_disagreement_opportunity": True,
            "minimum_triggered_rescue_opportunity": True,
            "minimum_rescue_precision": True, "minimum_net_corrected_errors": True,
            "known_accuracy_not_worse": True, "selective_error_not_worse": True,
            "mean_regret_not_worse": True, "false_known_acceptance_not_worse": False,
            "unsupported_precision_not_worse": True, "unsupported_recall_not_worse": True,
            "novel_evidence_exactly_unchanged": True,
        }
        result = classify_transfer(False, True, True, True, {
            "mechanism_gates": gates, "opportunity_sufficient": True, "mechanism_pass": False,
        })
        self.assertEqual(result["mechanism_status"], "rejected")
        self.assertFalse(result["preregister_sandboxed_typed_induction_feasibility"])


if __name__ == "__main__":
    unittest.main()
