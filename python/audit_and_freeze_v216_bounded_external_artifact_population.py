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
        "config": PROJECT_ROOT / "configs/v216-bounded-external-artifact-population.json",
        "plan": PROJECT_ROOT / "docs/v216-bounded-external-artifact-population-plan.md",
        "roadmap": PROJECT_ROOT / "docs/research-roadmap-after-v215.md",
        "protocol": PROJECT_ROOT / "python/v216_bounded_external_artifact_population.py",
        "tests": PROJECT_ROOT / "python/test_v216_bounded_external_artifact_population.py",
        "retrieval_worker": PROJECT_ROOT / "python/retrieve_v216_bounded_external_artifacts.py",
        "population_worker": PROJECT_ROOT / "python/build_v216_external_artifact_population.py",
        "runner": PROJECT_ROOT / "python/run_v216_bounded_external_artifact_population.py",
        "verifier": PROJECT_ROOT / "python/verify_and_freeze_v216_bounded_external_artifact_population_outcome.py",
        "auditor": PROJECT_ROOT / "python/audit_and_freeze_v216_bounded_external_artifact_population.py",
    }
    audit_path = PROJECT_ROOT / "outputs/v216-external-artifact-population/design-audit.json"
    lock_path = PROJECT_ROOT / "configs/v216-bounded-external-artifact-population-lock.json"
    output_root = PROJECT_ROOT / "outputs/v216-external-artifact-population"
    outcome_path = PROJECT_ROOT / "configs/v216-bounded-external-artifact-population-outcome-lock.json"
    if any(path.exists() for path in (audit_path, lock_path, output_root, outcome_path)):
        raise RuntimeError("V216 is already audited, frozen, retrieved, built, or outcome-frozen")
    config = json.loads(paths["config"].read_text())
    parent_path = PROJECT_ROOT / config["parentV215OutcomeLock"]
    parent = json.loads(parent_path.read_text())
    payloads = config["payloads"]
    retrieval = config["retrievalContract"]
    parser = config["parserDesign"]
    population = config["populationDesign"]
    split = config["splitDesign"]
    gates = config["populationGates"]
    exposure = config["priorExposure"]
    total_expected_bytes = sum(payload["expectedByteCount"] for payload in payloads)
    checks = {
        "V215_is_frozen_positive_and_authorizes_bounded_payload_design": bool(
            valid_lock(parent)
            and parent["outcome"]["passed"]
            and parent["outcome"]["branch"] == "BOUNDED_EXTERNAL_PAYLOAD_DESIGN_ELIGIBLE"
            and parent["authorization"]["design_bounded_external_payload_population"]
            and not parent["authorization"]["download_payload_without_separate_lock"]
            and not parent["authorization"]["open_protected_or_run_model"]
        ),
        "exact_three_role_separated_official_payloads_and_byte_budget_frozen": bool(
            len(payloads) == 3
            and len({payload["payloadId"] for payload in payloads}) == 3
            and {payload["role"] for payload in payloads}
            == {"OLDER_RETROSPECTIVE_SOURCE", "NEWER_RETROSPECTIVE_TARGET", "RDF_XML_PARSER_VALIDATION_CONTROL"}
            and all(urlparse(payload["url"]).scheme == "https" for payload in payloads)
            and {urlparse(payload["url"]).netloc for payload in payloads} == {"github.com", "www.w3.org"}
            and total_expected_bytes == 26924048
            and total_expected_bytes <= retrieval["maximumTotalPayloadBytes"] == gates["maximumTotalPayloadBytes"]
            and retrieval["maximumPayloadCount"] == 3
        ),
        "licenses_hashes_raw_preservation_and_no_substitution_are_frozen": bool(
            all(payload["license"] for payload in payloads)
            and all(payload["expectedByteCount"] > 0 and payload["rawPath"].startswith("outputs/v216-") for payload in payloads)
            and next(payload for payload in payloads if payload["payloadId"] == "UBERON_BASIC_2025_08_15")["declaredSha256"]
            == "9cb9db511e9d1d1d411902084eda676bdb6750f3e3009999cab30fe13836a452"
            and retrieval["exactlyOneAttemptPerPayload"]
            and retrieval["requireExactExpectedByteCount"]
            and retrieval["sha256EveryRawPayloadBeforeParsing"]
            and retrieval["noUnlistedNetworkRequest"]
            and retrieval["preserveRawFilesUnmodified"]
        ),
        "parser_and_semantic_claim_boundaries_are_explicit": bool(
            len(parser["logicalFields"]) == 5
            and not parser["w3cControlIsReasonerEvidence"]
            and not parser["assertedAxiomSignatureIsInferredSemanticEquivalence"]
            and "asserted fields" in config["claimBoundary"]
            and "not inferred OWL equivalence" in config["claimBoundary"]
        ),
        "public_truth_version_space_and_group_split_are_frozen": bool(
            population["oneRecordPerEligibleCurrentTerm"]
            and population["requiredCurrentDefinition"]
            and population["requiredCurrentAssertedLogicalAxiom"]
            and population["versionSpaceDefinition"].startswith("all_oracle_classes")
            and population["redactSourceIdentifiersFromPublicText"]
            and split["allRecordsWithSameObservationRemainTogether"]
            and split["protectedDownstreamMethodEvaluationCount"] == 0
            and split["protectedManualSemanticInspectionCount"] == 0
        ),
        "noncompensatory_payload_population_and_future_model_gates_are_frozen": bool(
            gates["requiredSuccessfulPayloadRetrievalRate"] == 1.0
            and gates["requiredRawHashCoverage"] == 1.0
            and gates["minimumTermsPerUberonRelease"] == 20000
            and gates["minimumEligibleGroupCount"] == 24
            and gates["minimumDistinctPrimaryChangeTypeCount"] == 2
            and gates["minimumDevelopmentGroupCount"] == 16
            and gates["minimumProtectedGroupCount"] == 8
            and gates["requiredVersionSpaceReconstructionAccuracy"] == 1.0
            and config["futureMethodEligibility"]["minimumPostDeterministicResidualDevelopmentGroups"] == 12
            and config["futureMethodEligibility"]["requiresIncrementalOracleClassRecallAtFixedCandidateBudget"]
            and config["futureMethodEligibility"]["requiresDecisionRelevantResidualBeforePlannerClaim"]
            and not config["futureMethodEligibility"]["apiModelRequired"]
        ),
        "prelock_metadata_exposure_disclosed_and_payload_unopened": bool(
            not exposure["blindPayloadDiscoveryClaim"]
            and exposure["officialReleaseAndArchiveMetadataInspectedBeforeLock"]
            and exposure["officialMetadataOrHeaderProbeCount"] > 0
            and exposure["payloadBodyByteReadCount"] == 0
            and exposure["ontologyTermOrAxiomRecordReadCount"] == 0
            and exposure["formalPopulationRecordCount"] == 0
        ),
        "pass_authority_is_narrow_and_all_outputs_absent": bool(
            config["decisionRule"]["passAuthorizesV217DesignOnly"]
            and not config["decisionRule"]["passAuthorizesProtectedMethodEvaluationOrModelRun"]
            and not config["decisionRule"]["passAuthorizesRegistrationMutationServiceActionOrExecution"]
            and all(path.is_file() for path in (*paths.values(), parent_path))
            and not output_root.exists()
            and not outcome_path.exists()
        ),
    }
    passed = all(checks.values())
    audit = {
        "schema_version": "216-bounded-external-artifact-population-design-audit",
        "experiment": config["experiment"],
        "passed": passed,
        "decision": "freeze_and_authorize_one_bounded_retrieval_and_population_build" if passed else "reject_V216_design",
        "checks": checks,
    }
    write_json(audit_path, audit)
    if not passed:
        print(json.dumps(audit, indent=2, sort_keys=True))
        raise SystemExit(1)
    dependencies = {**paths, "parent_V215_outcome": parent_path, "design_audit": audit_path}
    lock: dict[str, Any] = {
        "schema_version": "216-bounded-external-artifact-population-lock",
        "experiment": config["experiment"],
        "config_payload": config,
        "authorization": {
            "retrieve_exactly_three_frozen_payloads_once": True,
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

