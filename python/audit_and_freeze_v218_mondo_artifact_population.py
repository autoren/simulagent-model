#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from audit_and_freeze_v165_factored_ontology_identifiability_population import valid_lock
from audit_and_freeze_v167_exact_evidence_gathering_planner import payload_hash
from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def main() -> None:
    paths = {
        "config": PROJECT_ROOT / "configs/v218-mondo-artifact-population.json",
        "plan": PROJECT_ROOT / "docs/v218-mondo-artifact-population-plan.md",
        "roadmap": PROJECT_ROOT / "docs/research-roadmap-after-v217a.md",
        "protocol": PROJECT_ROOT / "python/v218_mondo_artifact_population.py",
        "tests": PROJECT_ROOT / "python/test_v218_mondo_artifact_population.py",
        "retrieval_worker": PROJECT_ROOT / "python/retrieve_v218_mondo_artifacts.py",
        "population_worker": PROJECT_ROOT / "python/build_v218_mondo_artifact_population.py",
        "runner": PROJECT_ROOT / "python/run_v218_mondo_artifact_population.py",
        "verifier": PROJECT_ROOT / "python/verify_and_freeze_v218_mondo_artifact_population_outcome.py",
        "auditor": PROJECT_ROOT / "python/audit_and_freeze_v218_mondo_artifact_population.py",
    }
    audit_path = PROJECT_ROOT / "outputs/v218-mondo-artifact-population/design-audit.json"
    lock_path = PROJECT_ROOT / "configs/v218-mondo-artifact-population-lock.json"
    output_root = PROJECT_ROOT / "outputs/v218-mondo-artifact-population"
    outcome_path = PROJECT_ROOT / "configs/v218-mondo-artifact-population-outcome-lock.json"
    if any(path.exists() for path in (audit_path, lock_path, output_root, outcome_path)):
        raise RuntimeError("V218 is already audited, frozen, retrieved, built, or outcome-frozen")
    config = json.loads(paths["config"].read_text())
    parent_path = PROJECT_ROOT / config["parentV217AOutcomeLock"]
    parent = json.loads(parent_path.read_text())
    metadata_path = PROJECT_ROOT / config["priorExposure"]["selectedReleaseMetadataSnapshot"]
    metadata = json.loads(metadata_path.read_text())
    frozen_assets = {
        asset["browser_download_url"]: (asset["size"], (asset.get("digest") or "").removeprefix("sha256:"))
        for release in metadata for asset in release.get("assets", [])
    }
    payloads = config["payloads"]
    retrieval = config["retrievalContract"]
    parser = config["parserDesign"]
    population = config["populationDesign"]
    gates = config["populationGates"]
    split = config["splitDesign"]
    exposure = config["priorExposure"]
    total_bytes = sum(payload["expectedByteCount"] for payload in payloads)
    expected_roles = {
        "OLDER_RETROSPECTIVE_SOURCE", "NEWER_RETROSPECTIVE_TARGET", "PAIR_CHANGED_TERM_CONTROL",
        "PAIR_NEW_TERM_CONTROL", "OLDER_CANDIDATE_STATUS_CONTROL", "NEWER_CANDIDATE_STATUS_CONTROL",
        "OLDER_SOURCE_PROVENANCE_CONTROL", "NEWER_SOURCE_PROVENANCE_CONTROL", "PUBLISHED_RELEASE_SUMMARY_CONTROL",
    }
    checks = {
        "V217A_is_verified_positive_and_authorizes_one_payload_design": bool(
            valid_lock(parent)
            and parent["outcome"]["verification_passed"]
            and parent["outcome"]["scientific_passed"]
            and parent["outcome"]["branch"] == "FRESH_INDEPENDENT_SOURCE_PAYLOAD_DESIGN_ELIGIBLE"
            and parent["outcome"]["selected_source_ids"] == ["MONDO"]
            and parent["authorization"]["design_one_fresh_source_payload_protocol"]
            and not parent["authorization"]["download_payload_open_V216_or_V213_protected_or_run_model"]
        ),
        "exact_nine_metadata_attested_assets_roles_and_closed_byte_budget_are_frozen": bool(
            len(payloads) == retrieval["maximumPayloadCount"] == gates["requiredPayloadCount"] == 9
            and len({payload["payloadId"] for payload in payloads}) == 9
            and {payload["role"] for payload in payloads} == expected_roles
            and all(urlparse(payload["url"]).scheme == "https" and urlparse(payload["url"]).netloc == "github.com" for payload in payloads)
            and all(payload["url"] in frozen_assets for payload in payloads)
            and all(frozen_assets[payload["url"]] == (payload["expectedByteCount"], payload["declaredSha256"]) for payload in payloads)
            and total_bytes == retrieval["expectedTotalPayloadBytes"] == 98957852
            and total_bytes <= retrieval["maximumTotalPayloadBytes"] == gates["maximumTotalPayloadBytes"] == 99000000
        ),
        "all_digests_licenses_raw_preservation_and_no_network_expansion_are_frozen": bool(
            all(len(payload["declaredSha256"]) == 64 and payload["license"] == "CC-BY-4.0" for payload in payloads)
            and all(payload["rawPath"].startswith("outputs/v218-") for payload in payloads)
            and retrieval["exactlyOneAttemptPerPayload"]
            and retrieval["requireExactExpectedByteCount"]
            and retrieval["sha256EveryRawPayloadBeforeParsing"]
            and retrieval["requireEveryDeclaredSha256"]
            and retrieval["noUnlistedNetworkRequest"]
            and retrieval["noRemoteImportResolution"]
            and retrieval["preserveRawFilesUnmodified"]
        ),
        "parser_event_and_semantic_claim_boundaries_are_explicit": bool(
            parser["duplicateTermIdPolicy"] == "FAIL"
            and parser["remoteImportPolicy"] == "FORBID_AND_DO_NOT_RESOLVE"
            and not parser["assertedStateIsInferredOWLEquivalence"]
            and len(config["eventDesign"]["eligibleEventTypes"]) == 10
            and "asserted fields and lifecycle states" in config["claimBoundary"]
            and "not inferred OWL equivalence" in config["claimBoundary"]
        ),
        "version_spaces_expressibility_witnesses_decisions_and_family_split_are_frozen": bool(
            population["oneFamilyPerUnionOfStableIdReplacementAndConsiderLinks"]
            and population["evidenceModes"] == ["VERSION_UNSPECIFIED", "CURRENT_RELEASE_DECLARED"]
            and population["versionSpaceDefinition"].startswith("all_old_current")
            and population["redactSourceIdentifiersFromPublicText"]
            and split["allRecordsAndLinkedIdentifiersInFamilyRemainTogether"]
            and split["protectedDownstreamMethodEvaluationCount"] == 0
            and split["protectedManualSemanticInspectionCount"] == 0
        ),
        "direct_noncompensatory_controls_population_and_future_model_gates_are_frozen": bool(
            gates["requiredSuccessfulPayloadRetrievalRate"] == 1.0
            and gates["requiredDeclaredDigestAccuracy"] == 1.0
            and gates["minimumParsedTermCountPerRelease"] == 1
            and "minimumTermsPerRelease" not in gates
            and gates["requiredNewTermControlAgreement"] == 1.0
            and gates["minimumEligibleConceptFamilyCount"] == 24
            and gates["minimumDistinctPrimaryEventTypeCount"] == 4
            and gates["minimumLifecycleEventFamilyCount"] == 3
            and gates["minimumAmbiguousUnspecifiedFamilyCount"] == 12
            and gates["requiredBoundaryWitnessCoverage"] == 1.0
            and gates["requiredDecisionConsequenceCoverage"] == 1.0
            and config["futureMethodEligibility"]["minimumPostDeterministicResidualDevelopmentGroups"] == 12
            and config["futureMethodEligibility"]["requiresIncrementalOracleClassRecallAtFixedCandidateBudget"]
            and not config["futureMethodEligibility"]["apiModelRequired"]
        ),
        "prelock_metadata_exposure_is_exact_and_payloads_protected_sources_remain_unopened": bool(
            not exposure["blindPayloadDiscoveryClaim"]
            and exposure["V217AMetadataInspectedBeforeLock"]
            and exposure["frozenPayloadURLCount"] == 9
            and file_sha256(metadata_path) == exposure["selectedReleaseMetadataSnapshotSha256"]
            and exposure["payloadBodyByteReadCount"] == 0
            and exposure["ontologyTermOrAxiomRecordReadCount"] == 0
            and exposure["formalPopulationRecordCount"] == 0
            and exposure["V216ProtectedReadCount"] == 0
            and exposure["V213ProtectedReadCount"] == 0
        ),
        "pass_authority_is_narrow_dependencies_exist_and_all_V218_outputs_are_absent": bool(
            config["decisionRule"]["passAuthorizesV219DesignOnly"]
            and not config["decisionRule"]["passAuthorizesProtectedMethodEvaluationOrModelRun"]
            and not config["decisionRule"]["passAuthorizesRegistrationMutationServiceActionOrExecution"]
            and all(path.is_file() for path in (*paths.values(), parent_path, metadata_path))
            and not output_root.exists()
            and not outcome_path.exists()
        ),
    }
    passed = all(checks.values())
    audit = {
        "schema_version": "218-mondo-artifact-population-design-audit",
        "experiment": config["experiment"],
        "passed": passed,
        "decision": "freeze_and_authorize_one_exact_bounded_retrieval_and_population_build" if passed else "reject_V218_design",
        "checks": checks,
    }
    write_json(audit_path, audit)
    if not passed:
        print(json.dumps(audit, indent=2, sort_keys=True))
        raise SystemExit(1)
    dependencies = {**paths, "parent_V217A_outcome": parent_path, "selected_release_metadata": metadata_path, "design_audit": audit_path}
    lock: dict[str, Any] = {
        "schema_version": "218-mondo-artifact-population-lock",
        "experiment": config["experiment"],
        "config_payload": config,
        "authorization": {
            "retrieve_exactly_nine_frozen_payloads_once": True,
            "build_one_role_separated_population": True,
            "structurally_verify_protected_partition_without_method_evaluation": True,
            "run_protected_method_or_model": False,
            "register_mutate_service_act_execute": False,
        },
    }
    for key, path in dependencies.items():
        lock[key] = str(path.relative_to(PROJECT_ROOT))
        lock[f"{key}_sha256"] = file_sha256(path)
    lock["lock_payload_sha256"] = payload_hash(lock)
    write_json(lock_path, lock)
    print(json.dumps(audit, indent=2, sort_keys=True))
    print(json.dumps({"lock": str(lock_path.relative_to(PROJECT_ROOT)), "sha256": file_sha256(lock_path)}, indent=2))


if __name__ == "__main__":
    main()
