#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
import re
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
        "config": PROJECT_ROOT / "configs/v220-fresh-mondo-artifact-population.json",
        "plan": PROJECT_ROOT / "docs/v220-fresh-mondo-artifact-population-plan.md",
        "roadmap": PROJECT_ROOT / "docs/research-roadmap-after-v219a.md",
        "protocol": PROJECT_ROOT / "python/v220_fresh_mondo_artifact_population.py",
        "inherited_protocol": PROJECT_ROOT / "python/v218_mondo_artifact_population.py",
        "tests": PROJECT_ROOT / "python/test_v220_fresh_mondo_artifact_population.py",
        "retrieval_worker": PROJECT_ROOT / "python/retrieve_v220_fresh_mondo_artifacts.py",
        "population_worker": PROJECT_ROOT / "python/build_v220_fresh_mondo_artifact_population.py",
        "runner": PROJECT_ROOT / "python/run_v220_fresh_mondo_artifact_population.py",
        "verifier": PROJECT_ROOT / "python/verify_and_freeze_v220_fresh_mondo_artifact_population_outcome.py",
        "auditor": PROJECT_ROOT / "python/audit_and_freeze_v220_fresh_mondo_artifact_population.py",
    }
    audit_path = PROJECT_ROOT / "outputs/v220-fresh-mondo-artifact-population/design-audit.json"
    lock_path = PROJECT_ROOT / "configs/v220-fresh-mondo-artifact-population-lock.json"
    output_root = PROJECT_ROOT / "outputs/v220-fresh-mondo-artifact-population"
    outcome_path = PROJECT_ROOT / "configs/v220-fresh-mondo-artifact-population-outcome-lock.json"
    if any(path.exists() for path in (audit_path, lock_path, output_root, outcome_path)):
        raise RuntimeError("V220 is already audited, frozen, retrieved, built, or outcome-frozen")
    config = json.loads(paths["config"].read_text())
    parent_path = PROJECT_ROOT / config["parentV219AOutcomeLock"]
    parent = json.loads(parent_path.read_text())
    selected = parent["outcome"]["selected_pair_assessments"]
    selected_roles = selected[0]["asset_roles"] if len(selected) == 1 else []
    expected_by_role = {row["role"]: row for row in selected_roles}
    payloads = config["payloads"]
    actual_by_role = {row["role"]: row for row in payloads}
    release = config["releaseSummaryControl"]
    snapshot_path = PROJECT_ROOT / release["snapshotPath"]
    retrieval = config["retrievalContract"]
    gates = config["populationGates"]
    exposure = config["priorExposure"]
    roles_exact = all(
        role in actual_by_role
        and actual_by_role[role]["url"] == expected["url"]
        and actual_by_role[role]["releaseTag"] == expected["release_tag"]
        and actual_by_role[role]["expectedByteCount"] == expected["byte_count"]
        and actual_by_role[role]["declaredSha256"] == expected["declared_sha256"]
        for role, expected in expected_by_role.items()
    )
    checks = {
        "V219A_verified_positive_selects_exactly_one_untouched_pair_and_authorizes_design": bool(
            valid_lock(parent)
            and parent["outcome"]["verification_passed"]
            and parent["outcome"]["scientific_passed"]
            and parent["outcome"]["branch"] == "UNTOUCHED_MONDO_PAIR_PAYLOAD_DESIGN_ELIGIBLE"
            and parent["outcome"]["selected_pair_ids"] == ["v2026-05-05__to__v2026-06-02"]
            and parent["authorization"]["design_one_untouched_pair_payload_protocol"]
            and not parent["authorization"]["retrieve_payload_or_evaluate_method"]
        ),
        "eight_payload_roles_URLs_bytes_and_published_digests_equal_V219A": bool(
            len(payloads) == len(selected_roles) == 8
            and set(actual_by_role) == set(expected_by_role)
            and roles_exact
            and len({row["payloadId"] for row in payloads}) == 8
            and all(urlparse(row["url"]).scheme == "https" and urlparse(row["url"]).netloc == "github.com" for row in payloads)
            and all(re.fullmatch(r"[0-9a-f]{64}", row["declaredSha256"]) for row in payloads)
            and all(row["license"] == "CC-BY-4.0" for row in payloads)
        ),
        "closed_retrieval_budget_and_one_attempt_integrity_contract_are_frozen": bool(
            sum(row["expectedByteCount"] for row in payloads) == retrieval["expectedTotalPayloadBytes"] == 96_095_320
            and retrieval["maximumPayloadCount"] == 8
            and retrieval["maximumTotalPayloadBytes"] == 97_000_000
            and retrieval["exactlyOneAttemptPerPayload"]
            and retrieval["requireExactExpectedByteCount"]
            and retrieval["sha256EveryRawPayloadBeforeParsing"]
            and retrieval["requireEveryDeclaredSha256"]
            and retrieval["noUnlistedNetworkRequest"]
            and retrieval["noRemoteImportResolution"]
            and retrieval["preserveRawFilesUnmodified"]
        ),
        "content_hashed_official_API_body_control_is_frozen_without_network_or_README": bool(
            snapshot_path.is_file()
            and file_sha256(snapshot_path) == release["snapshotSha256"]
            and release["newerReleaseTag"] == "v2026-06-02"
            and release["bodySha256"] == selected[0]["release_body_control"]["body_sha256"]
            and set(release["requiredCategories"]) == set(release["categoryPatterns"])
            and all(re.compile(pattern) for pattern in release["categoryPatterns"].values())
            and release["networkRequestCount"] == 0
            and not release["repositoryReadmeUsed"]
        ),
        "asserted_state_family_version_space_decision_and_group_split_design_is_frozen": bool(
            config["parserDesign"]["duplicateTermIdPolicy"] == "FAIL"
            and config["parserDesign"]["remoteImportPolicy"] == "FORBID_AND_DO_NOT_RESOLVE"
            and not config["parserDesign"]["assertedStateIsInferredOWLEquivalence"]
            and config["populationDesign"]["oneFamilyPerUnionOfStableIdReplacementAndConsiderLinks"]
            and config["populationDesign"]["evidenceModes"] == ["VERSION_UNSPECIFIED", "CURRENT_RELEASE_DECLARED"]
            and config["populationDesign"]["redactSourceIdentifiersFromPublicText"]
            and config["splitDesign"]["allRecordsAndLinkedIdentifiersInFamilyRemainTogether"]
        ),
        "event_strata_integrity_oracle_and_development_only_scoring_gates_are_noncompensatory": bool(
            gates["requiredPayloadCount"] == 8
            and gates["requiredDeclaredDigestAccuracy"] == 1.0
            and gates["requiredReleaseSummaryCategoryCoverage"] == 1.0
            and gates["minimumEligibleConceptFamilyCount"] == 24
            and gates["minimumAdditionEventFamilyCount"] == 24
            and gates["minimumTextChangeFamilyCount"] == 4
            and gates["minimumLifecycleEventFamilyCount"] == 3
            and gates["minimumMappingEventFamilyCount"] == 12
            and gates["minimumAmbiguousUnspecifiedFamilyCount"] == 12
            and gates["minimumDecisionContrastFamilyCount"] == 12
            and gates["requiredBoundaryWitnessCoverage"] == 1.0
            and gates["requiredDecisionConsequenceCoverage"] == 1.0
            and gates["oracleEvaluationScope"] == "DEVELOPMENT_ONLY_WITH_PROTECTED_FILES_HASHED_BUT_NOT_LOADED"
        ),
        "prelock_payload_V218_and_all_protected_exposure_is_zero": bool(
            not exposure["blindPayloadDiscoveryClaim"]
            and exposure["V219AMetadataInspectedBeforeLock"]
            and exposure["selectedPairId"] == "v2026-05-05__to__v2026-06-02"
            and exposure["openedV218ReleaseTags"] == ["v2026-07-06", "v2026-08-04"]
            and exposure["selectedPayloadBodyByteReadCount"] == 0
            and exposure["selectedOntologyTermOrAxiomRecordReadCount"] == 0
            and exposure["formalV220PopulationRecordCount"] == 0
            and exposure["V218DevelopmentRecordReadCount"] == 0
            and exposure["V218ProtectedRecordReadCount"] == 0
            and exposure["V216ProtectedReadCount"] == 0
            and exposure["V213ProtectedReadCount"] == 0
        ),
        "pass_authority_is_V221_design_only_and_all_dependencies_and_outputs_are_clean": bool(
            config["decisionRule"]["passAuthorizesV221DevelopmentOnlyDesign"]
            and not config["decisionRule"]["passAuthorizesProtectedMethodEvaluationOrModelRun"]
            and not config["decisionRule"]["passAuthorizesRegistrationMutationServiceActionOrExecution"]
            and all(path.is_file() for path in (*paths.values(), parent_path, snapshot_path))
            and not output_root.exists()
            and not outcome_path.exists()
        ),
    }
    passed = all(checks.values())
    audit = {
        "schema_version": "220-fresh-mondo-artifact-population-design-audit",
        "experiment": config["experiment"], "passed": passed,
        "decision": "freeze_and_authorize_one_exact_eight_payload_retrieval_and_population_build" if passed else "reject_V220_design",
        "checks": checks,
    }
    write_json(audit_path, audit)
    if not passed:
        print(json.dumps(audit, indent=2, sort_keys=True))
        raise SystemExit(1)
    dependencies = {
        **paths, "parent_V219A_outcome": parent_path,
        "release_metadata_snapshot": snapshot_path, "design_audit": audit_path,
    }
    lock: dict[str, Any] = {
        "schema_version": "220-fresh-mondo-artifact-population-lock",
        "experiment": config["experiment"], "config_payload": config,
        "authorization": {
            "retrieve_exactly_eight_frozen_payloads_once": True,
            "build_one_role_separated_population": True,
            "score_development_and_hash_but_do_not_load_protected": True,
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
