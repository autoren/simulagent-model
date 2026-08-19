from __future__ import annotations

import unittest

from v106_open_world_benchmark import prediction
from v164_local_residual_open_set_transfer import (
    combine_predictions,
    evaluate_quality_and_access_gates,
    render_residual_prompt,
)


class V164LocalResidualTests(unittest.TestCase):
    def test_prompt_exposes_no_identity_or_truth_fields(self) -> None:
        config = {
            "prompt": {
                "instruction": "classify",
            }
        }
        catalog = {"scenarios": ["iot"], "intents": []}
        rendered = render_residual_prompt(catalog, "turn it on", config)
        self.assertIn("turn it on", rendered)
        self.assertNotIn("record_id", rendered)
        self.assertNotIn("class_label", rendered)
        self.assertNotIn("truth", rendered)

    def test_combine_changes_only_residual_records(self) -> None:
        records = [{"record_id": "a"}, {"record_id": "b"}]
        consensus = {
            "a": prediction("KNOWN", 0.9, known_intent="iot::on"),
            "b": prediction("ABSTAIN", 0.0),
        }
        residual = {"b": prediction("UNSUPPORTED", 0.8)}
        combined = combine_predictions(records, residual, consensus)
        self.assertEqual(combined["a"], consensus["a"])
        self.assertEqual(combined["b"], residual["b"])

    def test_combine_rejects_unknown_residual_identity(self) -> None:
        with self.assertRaises(ValueError):
            combine_predictions(
                [{"record_id": "a"}],
                {"x": prediction("ABSTAIN", 0.0)},
                {"a": prediction("ABSTAIN", 0.0)},
            )

    def test_gate_logic_is_noncompensatory(self) -> None:
        metric = {
            "structured_response_validity": 1.0,
            "observed_status_macro_f1": 0.9,
            "observed_exact_decision_accuracy": 0.9,
            "known_exact_intent_accuracy": 0.9,
            "novel_exact_scenario_accuracy": 0.9,
            "false_known_acceptance_rate": 0.0,
            "confidence_ece_10_bin": 0.05,
            "top_confidence_80_percent_error": 0.1,
            "mean_regret": 0.2,
            "per_status": {
                "NOVEL": {"recall": 0.9, "precision": 0.9},
                "UNSUPPORTED": {"recall": 0.9, "precision": 0.9},
            },
        }
        aggregate = {
            "fixture_count": 76,
            "residual_metrics": dict(metric),
            "hybrid_metrics": dict(metric),
            "controlled_missing_observation_abstention_accuracy": 1.0,
            "true_hypothesis_retention": 1.0,
        }
        quality = {
            "minimumStructuredResponseValidity": 0.98,
            "minimumResidualStatusMacroF1": 0.65,
            "minimumResidualExactDecisionAccuracy": 0.6,
            "minimumResidualKnownExactIntentAccuracy": 0.65,
            "minimumResidualNovelStatusRecall": 0.7,
            "minimumResidualNovelStatusPrecision": 0.7,
            "minimumResidualNovelExactScenarioAccuracy": 0.7,
            "minimumResidualUnsupportedStatusRecall": 0.8,
            "minimumResidualUnsupportedStatusPrecision": 0.8,
            "maximumResidualFalseKnownAcceptanceRate": 0.1,
            "maximumResidualConfidenceECE": 0.15,
            "maximumResidualTopConfidence80PercentError": 0.3,
            "maximumResidualMeanDecisionRegret": 1.0,
            "minimumCombinedHybridExactDecisionAccuracy": 0.65,
            "maximumCombinedHybridFalseKnownAcceptanceRate": 0.05,
            "maximumCombinedHybridMeanDecisionRegret": 0.85,
            "minimumCombinedRegretImprovementOverFrozenConsensus": 0.1,
            "requiredControlledMissingObservationAbstentionAccuracy": 1.0,
            "requiredTrueHypothesisRetention": 1.0,
        }
        limits = {
            "requiredResidualFixtureCount": 76,
            "maximumDevelopmentLanguageReadCount": 1,
            "maximumProtectedLanguageReadCount": 0,
            "maximumManualUtteranceInspectionCount": 0,
            "maximumManualRawResponseInspectionCount": 0,
            "maximumModelLoadCount": 1,
            "maximumModelGenerationCount": 76,
            "maximumRetryCount": 0,
            "maximumLLMAPICallCount": 0,
            "maximumAdapterTrainingRunCount": 0,
            "maximumRealServiceCallCount": 0,
            "maximumExternalSideEffectCount": 0,
            "maximumActualExecutionCount": 0,
        }
        access = {
            "development_language_read_count": 1,
            "protected_language_read_count": 0,
            "manual_utterance_inspection_count": 0,
            "manual_raw_response_inspection_count": 0,
            "model_load_count": 1,
            "model_generation_count": 76,
            "retry_count": 0,
            "LLM_API_call_count": 0,
            "adapter_training_run_count": 0,
            "real_service_call_count": 0,
            "external_side_effect_count": 0,
            "actual_execution_count": 0,
        }
        checks = evaluate_quality_and_access_gates(
            aggregate, 0.953125, access,
            {"qualityGates": quality, "accessGates": limits},
        )
        self.assertTrue(all(checks.values()))
        aggregate["residual_metrics"]["false_known_acceptance_rate"] = 0.2
        checks = evaluate_quality_and_access_gates(
            aggregate, 0.953125, access,
            {"qualityGates": quality, "accessGates": limits},
        )
        self.assertFalse(checks["residual_false_known_acceptance_rate"])


if __name__ == "__main__":
    unittest.main()
