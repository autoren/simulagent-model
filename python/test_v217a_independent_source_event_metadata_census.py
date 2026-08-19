from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from v217a_independent_source_event_metadata_census import audit_census, score_census


class V217AIndependentSourceEventCensusTests(unittest.TestCase):
    def test_priority_selects_lowest_number_eligible_source(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            snapshot = root / "meta.json"
            snapshot.write_text("{}")
            import hashlib
            sha = hashlib.sha256(snapshot.read_bytes()).hexdigest()
            dimensions = ["d"]
            config = {
                "sourceUnits": [
                    {"unitId": "A", "selectionPriority": 2, "urls": ["https://x/a"]},
                    {"unitId": "B", "selectionPriority": 1, "urls": ["https://x/b"]},
                ],
                "dimensions": dimensions,
                "eventCategories": ["TEXT", "LIFE", "OTHER"],
                "textEventCategories": ["TEXT"],
                "lifecycleOrMappingEventCategories": ["LIFE"],
                "ambiguityIndicators": ["AMB"],
                "eligibilityRequirements": {
                    "requiredDimensions": dimensions,
                    "minimumExactHistoricalReleaseCount": 2,
                    "minimumDocumentedEventCategoryCount": 3,
                    "requireTextEventCategory": True,
                    "requireLifecycleOrMappingEventCategory": True,
                    "minimumAmbiguityIndicatorCount": 1,
                    "maximumSingleAssetBytes": 10,
                    "maximumTwoReleasePayloadBytes": 20,
                },
            }
            manifest = {"attempts": [
                {"unit_id": unit, "url": f"https://x/{unit.lower()}", "success": True, "snapshot_path": str(snapshot), "sha256": sha, "byte_count": 2}
                for unit in ["A", "B"]
            ]}
            source_units = []
            for unit in ["A", "B"]:
                url = f"https://x/{unit.lower()}"
                source_units.append({
                    "unit_id": unit,
                    "assessments": {"d": True},
                    "assessment_evidence_urls": {"d": [url]},
                    "documented_event_categories": ["TEXT", "LIFE", "OTHER"],
                    "event_category_evidence_urls": {key: [url] for key in ["TEXT", "LIFE", "OTHER"]},
                    "ambiguity_indicators": ["AMB"],
                    "ambiguity_indicator_evidence_urls": {"AMB": [url]},
                    "historical_releases": [
                        {"release_id": "r1", "metadata_url": url, "bounded_assets": [{"url": "https://a/r1.obo", "byte_count": 5, "format": "OBO"}]},
                        {"release_id": "r2", "metadata_url": url, "bounded_assets": [{"url": "https://a/r2.obo", "byte_count": 5, "format": "OBO"}]},
                    ],
                })
            evidence = {
                "source_units": source_units,
                "recommended_source_ids": ["B"],
                "recommendation": {"retrospective_artifact_not_new_speaker_intent": True},
            }
            metrics = score_census(manifest, evidence, config, root)
        self.assertEqual(["B"], metrics["expected_selected_source_ids"])
        self.assertEqual(1.0, metrics["selection_priority_correctness"])

    def test_negative_scientific_audit_is_a_valid_branch(self) -> None:
        config = {
            "censusGates": {
                "requiredSourceUnitCount": 1, "requiredFrozenURLAttemptCount": 1, "requiredURLAccountingRate": 1.0,
                "requiredSuccessfulSnapshotHashCoverage": 1.0, "requiredAssessmentDimensionCoverage": 1.0,
                "requiredTrueClaimEvidenceCoverage": 1.0, "minimumEligibleSourceCount": 1,
                "maximumSelectedSourceCount": 1, "requiredSelectionPriorityCorrectness": 1.0,
                "requiredRetrospectiveNotSpeakerIntentBoundary": True, "requiredFiniteMetrics": True,
            },
            "accessGates": {
                "requiredMetadataCaptureRunCount": 1, "requiredMetadataCensusRunCount": 1,
                "maximumCandidatePayloadDownloadCount": 0, "maximumV216ProtectedAccessCount": 0,
                "maximumV213ProtectedAccessCount": 0, "maximumModelLoadCount": 0,
                "maximumModelGenerationCount": 0, "maximumModelAPICallCount": 0,
                "maximumTrainingRunCount": 0, "maximumOntologyRegistrationCount": 0,
                "maximumTrustedStateMutationCount": 0, "maximumServiceActionCount": 0,
                "maximumExternalSideEffectCountBeyondReadOnlyMetadata": 0, "maximumActualExecutionCount": 0,
            },
            "decisionRule": {
                "ifEveryIntegrityEligibilitySelectionAndAccessGatePasses": "positive",
                "otherwise": "negative",
            },
        }
        metrics = {
            "source_unit_count": 1, "frozen_url_attempt_count": 1, "recorded_url_attempt_count": 1,
            "url_accounting_rate": 1.0, "unexpected_url_attempt_count": 0,
            "successful_snapshot_hash_coverage": 1.0, "assessment_dimension_coverage": 1.0,
            "true_claim_evidence_coverage": 1.0, "eligible_source_count": 0,
            "selected_source_count": 0, "selection_priority_correctness": 1.0,
            "retrospective_not_speaker_intent_boundary": True, "total_term_count_used_for_eligibility": False,
            "finite_metrics": True,
        }
        access = {key: 0 for key in [
            "candidate_payload_download_count", "v216_protected_access_count", "v213_protected_access_count",
            "model_load_count", "model_generation_count", "model_api_call_count", "training_run_count",
            "ontology_registration_count", "trusted_state_mutation_count", "service_action_count",
            "external_side_effect_count_beyond_read_only_metadata", "actual_execution_count",
        ]}
        access.update({"metadata_capture_run_count": 1, "metadata_census_run_count": 1})
        audit = audit_census(metrics, access, config)
        self.assertFalse(audit["passed"])
        self.assertEqual("STOP_EXTERNAL_RETROSPECTIVE_SOURCE_BRANCH", audit["branch"])


if __name__ == "__main__":
    unittest.main()

