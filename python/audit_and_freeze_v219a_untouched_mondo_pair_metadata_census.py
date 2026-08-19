#!/usr/bin/env python3
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from audit_and_freeze_v165_factored_ontology_identifiability_population import valid_lock
from audit_and_freeze_v167_exact_evidence_gathering_planner import payload_hash
from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def main() -> None:
    paths = {
        "config": PROJECT_ROOT / "configs/v219a-untouched-mondo-pair-metadata-census.json",
        "plan": PROJECT_ROOT / "docs/v219a-untouched-mondo-pair-metadata-census-plan.md",
        "roadmap": PROJECT_ROOT / "docs/research-roadmap-after-v218.md",
        "protocol": PROJECT_ROOT / "python/v219a_untouched_mondo_pair_metadata_census.py",
        "tests": PROJECT_ROOT / "python/test_v219a_untouched_mondo_pair_metadata_census.py",
        "runner": PROJECT_ROOT / "python/run_v219a_untouched_mondo_pair_metadata_census.py",
        "verifier": PROJECT_ROOT / "python/verify_and_freeze_v219a_untouched_mondo_pair_metadata_census_outcome.py",
        "auditor": PROJECT_ROOT / "python/audit_and_freeze_v219a_untouched_mondo_pair_metadata_census.py",
    }
    audit_path = PROJECT_ROOT / "outputs/v219a-untouched-mondo-pair-metadata-census/design-audit.json"
    lock_path = PROJECT_ROOT / "configs/v219a-untouched-mondo-pair-metadata-census-lock.json"
    output_root = PROJECT_ROOT / "outputs/v219a-untouched-mondo-pair-metadata-census"
    outcome_path = PROJECT_ROOT / "configs/v219a-untouched-mondo-pair-metadata-census-outcome-lock.json"
    if any(path.exists() for path in (audit_path, lock_path, output_root, outcome_path)):
        raise RuntimeError("V219A is already audited, frozen, run, or outcome-frozen")
    config = json.loads(paths["config"].read_text())
    parent_path = PROJECT_ROOT / config["parentV218OutcomeLock"]
    parent = json.loads(parent_path.read_text())
    snapshot_path = PROJECT_ROOT / config["evidenceSource"]["path"]
    enumeration = config["pairEnumeration"]
    assets = config["requiredAssetRoles"]
    requirements = config["assetRequirements"]
    body = config["releaseBodyControl"]
    gates = config["censusGates"]
    exposure = config["priorExposure"]
    expected_roles = {
        "OLDER_RETROSPECTIVE_SOURCE", "NEWER_RETROSPECTIVE_TARGET",
        "PAIR_CHANGED_TERM_CONTROL", "PAIR_NEW_TERM_CONTROL",
        "OLDER_CANDIDATE_STATUS_CONTROL", "NEWER_CANDIDATE_STATUS_CONTROL",
        "OLDER_SOURCE_PROVENANCE_CONTROL", "NEWER_SOURCE_PROVENANCE_CONTROL",
    }
    checks = {
        "V218_verified_negative_authorizes_no_deterministic_or_model_continuation": bool(
            valid_lock(parent)
            and parent["outcome"]["verification_passed"]
            and not parent["outcome"]["scientific_passed"]
            and parent["outcome"]["branch"] == "NEGATIVE_MONDO_PAYLOAD_CONTROL_OR_POPULATION_FEASIBILITY"
            and not parent["authorization"]["design_V219_deterministic_controls"]
            and not parent["authorization"]["open_protected_or_run_model"]
        ),
        "one_exact_preexisting_snapshot_is_frozen_with_network_disabled": bool(
            snapshot_path.is_file()
            and file_sha256(snapshot_path) == config["evidenceSource"]["sha256"]
            and config["evidenceSource"]["expectedReleaseCount"] == 5
            and not config["evidenceSource"]["networkRequestAuthorized"]
        ),
        "full_order_adjacency_exclusions_and_single_newest_selection_are_frozen": bool(
            enumeration["releaseOrder"] == "published_at_descending_then_tag_name_ascending"
            and enumeration["adjacencyDefinition"] == "consecutive_entries_in_the_full_frozen_release_order_before_exclusion"
            and enumeration["excludedReleaseTags"] == ["v2026-07-06", "v2026-08-04"]
            and enumeration["keepPairOnlyIfBothReleaseTagsAreUnexcluded"]
            and enumeration["selectAtMostOne"]
            and enumeration["selectFirstEligiblePair"]
            and enumeration["expectedUntouchedAdjacentPairCount"] == 2
        ),
        "eight_exact_asset_roles_digests_and_closed_byte_bounds_are_frozen": bool(
            len(assets) == requirements["requiredAssetRoleCount"] == 8
            and {asset["role"] for asset in assets} == expected_roles
            and sum(asset["assetName"] == "mondo-base.obo" for asset in assets) == 2
            and {asset["assetName"] for asset in assets if asset["role"].startswith("PAIR_")} == {
                "mondo_release_diff_changed_terms.tsv", "mondo_release_diff_new_terms.tsv"
            }
            and requirements["requireHttpsDownloadUrl"]
            and requirements["requirePositiveIntegerByteCount"]
            and requirements["requirePublishedSha256Digest"]
            and requirements["maximumSingleAssetBytes"] == 50_000_000
            and requirements["maximumPairPayloadBytes"] == 99_000_000
            and not requirements["allowAssetSubstitution"]
            and not requirements["allowLinkExpansion"]
        ),
        "official_API_body_hash_and_four_release_categories_are_frozen": bool(
            body["source"] == "body_field_of_newer_release_object_in_frozen_API_snapshot"
            and body["storeExactBodySha256InEvidence"]
            and set(body["requiredCategories"]) == {
                "ADDITION", "SYNONYM_OR_LABEL", "TEXT_DEFINITION", "OBSOLETION_OR_REPLACEMENT"
            }
            and set(body["categoryPatterns"]) == set(body["requiredCategories"])
            and all(re.compile(pattern) for pattern in body["categoryPatterns"].values())
            and not body["repositoryReadmeIsReleaseSummaryEvidence"]
        ),
        "noncompensatory_census_and_zero_access_gates_are_frozen": bool(
            gates["requiredEvidenceSnapshotHashAccuracy"] == 1.0
            and gates["requiredUntouchedAdjacentPairCount"] == 2
            and gates["requiredOriginalAdjacencyAccuracy"] == 1.0
            and gates["requiredAssetRoleCoverage"] == 1.0
            and gates["requiredAssetDigestCoverage"] == 1.0
            and gates["requiredReleaseBodyHashCoverage"] == 1.0
            and gates["requiredReleaseBodyCategoryCoverage"] == 1.0
            and gates["minimumEligiblePairCount"] == 1
            and gates["maximumSelectedPairCount"] == 1
            and all(value == 0 for key, value in config["accessGates"].items() if key.startswith("maximum"))
        ),
        "prelock_exposure_excludes_all_new_payload_and_protected_records": bool(
            not exposure["blindReleaseMetadataDiscoveryClaim"]
            and exposure["V217AReleaseMetadataPreviouslyInspected"]
            and exposure["V218OpenedReleaseTags"] == ["v2026-07-06", "v2026-08-04"]
            and exposure["newPayloadBodyByteReadCount"] == 0
            and exposure["newOntologyTermOrAxiomRecordReadCount"] == 0
            and exposure["V218DevelopmentRecordReadCount"] == 0
            and exposure["V218ProtectedRecordReadCount"] == 0
            and exposure["V216ProtectedReadCount"] == 0
            and exposure["V213ProtectedReadCount"] == 0
        ),
        "pass_authority_is_one_payload_design_only_and_all_outputs_are_absent": bool(
            config["decisionRule"]["passAuthorizesOnePayloadDesignOnly"]
            and not config["decisionRule"]["passAuthorizesPayloadRetrievalOrMethodEvaluation"]
            and not config["decisionRule"]["passAuthorizesProtectedAccessOrModelRun"]
            and not config["decisionRule"]["passAuthorizesRegistrationMutationServiceActionOrExecution"]
            and all(path.is_file() for path in (*paths.values(), parent_path, snapshot_path))
            and not output_root.exists()
            and not outcome_path.exists()
        ),
    }
    passed = all(checks.values())
    audit = {
        "schema_version": "219a-untouched-mondo-pair-metadata-census-design-audit",
        "experiment": config["experiment"],
        "passed": passed,
        "decision": "freeze_and_authorize_one_local_metadata_only_untouched_pair_census" if passed else "reject_V219A_design",
        "checks": checks,
    }
    write_json(audit_path, audit)
    if not passed:
        print(json.dumps(audit, indent=2, sort_keys=True))
        raise SystemExit(1)
    dependencies = {
        **paths,
        "parent_V218_outcome": parent_path,
        "evidence_snapshot": snapshot_path,
        "design_audit": audit_path,
    }
    lock: dict[str, Any] = {
        "schema_version": "219a-untouched-mondo-pair-metadata-census-lock",
        "experiment": config["experiment"],
        "config_payload": config,
        "authorization": {
            "run_one_local_metadata_census": True,
            "select_at_most_one_untouched_pair": True,
            "design_one_pair_payload_protocol_after_positive": True,
            "retrieve_payload_or_evaluate_method": False,
            "open_protected_or_run_model": False,
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
