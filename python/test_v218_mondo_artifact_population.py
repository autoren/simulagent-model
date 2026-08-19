from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from v218_mondo_artifact_population import (
    audit_population,
    build_population_records,
    parse_obo_text,
    parse_tsv,
    redact_source_ids,
    release_summary_control,
)


def synthetic_config() -> dict:
    return {
        "experiment": "synthetic_v218",
        "parserDesign": {
            "logicalFields": ["is_a", "relationship", "intersection_of", "equivalent_to", "disjoint_from"],
            "mappingFields": ["xref"],
        },
        "eventDesign": {
            "primaryEventPrecedence": [
                "REPLACEMENT_CHANGED", "OBSOLETION_CHANGED", "OBSOLETION_CANDIDATE_STATUS_CHANGED",
                "ADDED", "REMOVED", "NAME_CHANGED", "DEFINITION_CHANGED", "SYNONYM_CHANGED",
                "LOGICAL_AXIOM_CHANGED", "MAPPING_CHANGED",
            ]
        },
        "populationDesign": {"evidenceModes": ["VERSION_UNSPECIFIED", "CURRENT_RELEASE_DECLARED"]},
    }


class V218MondoArtifactPopulationTests(unittest.TestCase):
    def test_lifecycle_family_preserves_version_space_and_changes_decision(self) -> None:
        older = parse_obo_text(
            """
[Term]
id: MONDO:0001
name: old disease
def: "old definition" []
is_a: MONDO:9999 ! parent

[Term]
id: MONDO:0002
name: target disease
def: "target definition" []
is_a: MONDO:9999 ! parent
"""
        )
        newer = parse_obo_text(
            """
[Term]
id: MONDO:0001
name: obsolete old disease
def: "old definition" []
is_obsolete: true
replaced_by: MONDO:0002

[Term]
id: MONDO:0002
name: target disease
def: "target definition" []
is_a: MONDO:9999 ! parent

[Term]
id: MONDO:0003
name: new disease
def: "new definition" []
is_a: MONDO:9999 ! parent
"""
        )
        public, truth, manifest = build_population_records(
            older, newer, {"MONDO:0001"}, set(), synthetic_config()
        )
        self.assertEqual(2, manifest["eligible_concept_family_count"])
        lifecycle = [record for record in truth if "MONDO:0001" in record["source_concept_ids"]]
        self.assertEqual(2, len(lifecycle))
        unspecified = next(record for record in lifecycle if record["evidence_state"] == "AMBIGUOUS")
        declared = next(record for record in lifecycle if record["correct_decision"] == "FOLLOW_ASSERTED_REPLACEMENT")
        self.assertGreaterEqual(len(unspecified["candidate_class_ids"]), 2)
        self.assertEqual(unspecified["replacement_target_class_ids"], declared["candidate_class_ids"])
        self.assertTrue(all("MONDO:" not in json.dumps(record) for record in public))

    def test_controls_parse_without_interpreting_semantic_truth(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tsv = root / "control.tsv"
            tsv.write_text("id\tlabel\nMONDO:0001\tone\nMONDO:0002\ttwo\n")
            readme = root / "README.md"
            readme.write_text(
                "New terms\nTerms renamed\nText definitions changed\nTerms obsoleted with replacement\n"
            )
            parsed = parse_tsv(tsv)
            summary = release_summary_control(readme)
        self.assertTrue(parsed["parse_success"])
        self.assertTrue(parsed["mondo_id_unique"])
        self.assertEqual(["MONDO:0001", "MONDO:0002"], parsed["mondo_ids"])
        self.assertEqual(4, len(summary["categories"]))
        self.assertEqual("request [SOURCE_ID]", redact_source_ids("request MONDO:1234"))

    def test_negative_population_is_frozen_by_noncompensatory_gate(self) -> None:
        config = {
            "populationGates": {
                "requiredPayloadCount": 1, "requiredSuccessfulPayloadRetrievalRate": 1.0,
                "requiredRawHashCoverage": 1.0, "requiredExpectedByteCountAccuracy": 1.0,
                "requiredDeclaredDigestAccuracy": 1.0, "maximumTotalPayloadBytes": 10,
                "minimumParsedTermCountPerRelease": 1, "requiredUniqueTermIdRate": 1.0,
                "requiredRemoteImportResolutionCount": 0, "requiredTabularControlParseRate": 1.0,
                "requiredNewTermControlAgreement": 1.0, "requiredChangedTermControlPrecision": 1.0,
                "requiredReleaseSummaryCategoryCoverage": 1.0, "minimumEligibleConceptFamilyCount": 2,
                "minimumEligibleRecordCount": 4, "minimumDistinctPrimaryEventTypeCount": 1,
                "minimumTextChangeFamilyCount": 1, "minimumLifecycleEventFamilyCount": 1,
                "minimumAmbiguousUnspecifiedFamilyCount": 1, "minimumDecisionContrastFamilyCount": 1,
                "minimumDevelopmentGroupCount": 1, "minimumProtectedGroupCount": 1,
                "requiredSemanticStateReconstructionAccuracy": 1.0,
                "requiredVersionSpaceReconstructionAccuracy": 1.0, "requiredBoundaryWitnessCoverage": 1.0,
                "requiredDecisionConsequenceCoverage": 1.0, "maximumCrossSplitGroupOverlapCount": 0,
                "maximumDuplicateCaseIdCount": 0, "maximumPublicSourceIdentifierLeakageCount": 0,
                "requiredPublicTruthCaseAlignment": True, "requiredSplitManifestExact": True,
                "requiredFiniteMetrics": True,
            },
            "accessGates": {
                "requiredBoundedRetrievalRunCount": 1, "requiredPopulationBuildRunCount": 1,
                "maximumUnlistedNetworkRequestCount": 0, "maximumPayloadCount": 1,
                "maximumRemoteImportResolutionCount": 0, "maximumV216ProtectedAccessCount": 0,
                "maximumV213ProtectedAccessCount": 0, "maximumProtectedDownstreamMethodEvaluationCount": 0,
                "maximumProtectedManualSemanticInspectionCount": 0, "maximumModelLoadCount": 0,
                "maximumModelGenerationCount": 0, "maximumModelAPICallCount": 0,
                "maximumTrainingRunCount": 0, "maximumOntologyRegistrationCount": 0,
                "maximumTrustedStateMutationCount": 0, "maximumServiceActionCount": 0,
                "maximumExternalSideEffectCountBeyondReadOnlyRetrieval": 0, "maximumActualExecutionCount": 0,
            },
            "decisionRule": {
                "ifEveryPayloadControlPopulationIntegrityAndAccessGatePasses": "positive",
                "otherwise": "negative",
            },
        }
        metrics = {
            "payload_count": 1, "attempt_count": 1, "exact_payload_id_accounting": True,
            "successful_payload_retrieval_rate": 1.0, "raw_hash_coverage": 1.0,
            "expected_byte_count_accuracy": 1.0, "declared_digest_accuracy": 1.0,
            "total_payload_bytes": 1, "older_term_count": 1, "newer_term_count": 1,
            "unique_term_id_rate": 1.0, "remote_import_resolution_count": 0,
            "tabular_control_parse_rate": 1.0, "new_term_control_agreement": 1.0,
            "changed_term_control_precision": 1.0, "release_summary_category_coverage": 1.0,
            "eligible_concept_family_count": 1, "eligible_record_count": 2,
            "development_group_count": 1, "protected_group_count": 0,
            "distinct_primary_event_type_count": 1, "text_change_family_count": 1,
            "lifecycle_event_family_count": 1, "ambiguous_unspecified_family_count": 1,
            "decision_contrast_family_count": 1, "semantic_state_reconstruction_accuracy": 1.0,
            "version_space_reconstruction_accuracy": 1.0, "boundary_witness_coverage": 1.0,
            "decision_consequence_coverage": 1.0, "cross_split_group_overlap_count": 0,
            "duplicate_case_id_count": 0, "public_source_identifier_leakage_count": 0,
            "public_truth_case_alignment": True, "split_manifest_exact": True, "finite_metrics": True,
        }
        access = {key: 0 for key in [
            "unlisted_network_request_count", "remote_import_resolution_count", "v216_protected_access_count",
            "v213_protected_access_count", "protected_downstream_method_evaluation_count",
            "protected_manual_semantic_inspection_count", "model_load_count", "model_generation_count",
            "model_api_call_count", "training_run_count", "ontology_registration_count",
            "trusted_state_mutation_count", "service_action_count",
            "external_side_effect_count_beyond_read_only_retrieval", "actual_execution_count",
        ]}
        access.update({"bounded_retrieval_run_count": 1, "population_build_run_count": 1, "payload_count": 1})
        audit = audit_population(metrics, access, config)
        self.assertFalse(audit["passed"])
        self.assertEqual("NEGATIVE_MONDO_PAYLOAD_CONTROL_OR_POPULATION_FEASIBILITY", audit["branch"])


if __name__ == "__main__":
    unittest.main()
