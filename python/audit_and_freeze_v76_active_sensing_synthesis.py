#!/usr/bin/env python3
"""Audit V71-V75 evidence and freeze the V76 discovery-clean source protocol."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT


SOURCES = {
    "V71": "configs/v71-sensor-codebook-development-outcome-lock.json",
    "V72-oracle": "configs/v72-active-sensing-oracle-outcome-lock.json",
    "V72-external": "configs/v72-active-sensing-development-outcome-lock.json",
    "V73": "configs/v73-active-sensing-structural-outcome-lock.json",
    "V74": "configs/v74-active-sensing-development-outcome-lock.json",
    "V75": "configs/v75-active-sensing-confirmation-outcome-lock.json",
}
SYNTHESIS_CONFIG = "configs/v76-active-sensing-evidence-synthesis.json"
SYNTHESIS_DOCUMENT = "docs/v76-active-sensing-evidence-synthesis.md"
CENSUS_CONFIG = "configs/v76-discovery-clean-source-census.json"
CENSUS_PLAN = "docs/v76-discovery-clean-source-census-plan.md"
RESEARCH_DIRECTION = "docs/research-direction.md"
AUDITOR = "python/audit_and_freeze_v76_active_sensing_synthesis.py"
AUDIT = "outputs/v76-active-sensing-synthesis/audit.json"
LOCK = "configs/v76-active-sensing-synthesis-lock.json"


def canonical_hash(value: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def load_json(relative_path: str) -> dict[str, Any]:
    return json.loads((PROJECT_ROOT / relative_path).read_text())


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def close(actual: float, expected: float) -> bool:
    return math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-12)


def lock_payload_is_valid(lock: dict[str, Any]) -> bool:
    payload = {key: value for key, value in lock.items() if key != "lock_payload_sha256"}
    return canonical_hash(payload) == lock["lock_payload_sha256"]


def bound_file(lock: dict[str, Any], path_key: str, hash_key: str) -> bool:
    relative_path = lock.get(path_key)
    expected_hash = lock.get(hash_key)
    if not isinstance(relative_path, str) or not isinstance(expected_hash, str):
        return False
    path = PROJECT_ROOT / relative_path
    return path.is_file() and file_sha256(path) == expected_hash


def source_chain_is_bound(stage: str, lock: dict[str, Any]) -> bool:
    required = {
        "V71": (("result", "result_sha256"), ("report", "report_sha256"), ("outcome_audit", "outcome_audit_sha256")),
        "V72-oracle": (("result", "result_sha256"), ("report", "report_sha256"), ("outcome_audit", "outcome_audit_sha256")),
        "V72-external": (("result", "result_sha256"), ("report", "report_sha256"), ("outcome_audit", "outcome_audit_sha256")),
        "V73": (("structural_audit", "structural_audit_sha256"), ("report", "report_sha256"), ("outcome_audit", "outcome_audit_sha256")),
        "V74": (("result", "result_sha256"), ("report", "report_sha256"), ("outcome_audit", "outcome_audit_sha256")),
        "V75": (("result", "result_sha256"), ("report", "report_sha256"), ("outcome_audit", "outcome_audit_sha256")),
    }
    return lock_payload_is_valid(lock) and all(
        bound_file(lock, path_key, hash_key) for path_key, hash_key in required[stage]
    )


def audit_outcomes(locks: dict[str, dict[str, Any]]) -> dict[str, bool]:
    v71 = locks["V71"]["outcome"]
    v72o = load_json(locks["V72-oracle"]["result"])
    v72e = load_json(locks["V72-external"]["result"])
    v73 = load_json(locks["V73"]["structural_audit"])
    v74 = load_json(locks["V74"]["result"])
    v75 = load_json(locks["V75"]["result"])

    v72_positive = v72o["fixtures"]["positive"]
    v72_row = v72e["row"]
    v74_model = v74["model"]
    v75_model = v75["model"]

    no_human_model_or_adapter = bool(
        all(
            result["access"]["human_record_access_count"] == 0
            and result["access"]["model_forward_pass_count"] == 0
            and result["access"]["adapter_training_run_count"] == 0
            for result in (v72o, v72e, v73, v74, v75)
        )
    )

    return {
        "all_source_outcome_chains_are_hash_bound": all(
            source_chain_is_bound(stage, locks[stage]) for stage in SOURCES
        ),
        "v71_is_clean_zero_regret_negative_boundary": bool(
            not v71["passed_development_gates"]
            and v71["development_models"] == 3
            and v71["development_records"] == 21
            and close(v71["maximum_normalized_MAP_regret"], 0.0)
            and v71["models_with_exact_BA_MAP_root_action_disagreement"] == 0
            and v71["models_with_material_MAP_regret"] == 0
            and v71["models_with_material_posterior_sampling_regret"] == 0
            and v71["protected_confirmation_policy_value_count"] == 0
        ),
        "v72_oracle_validates_mechanism_only": bool(
            v72o["passed"]
            and v72o["claim_boundary"] == "engineered mechanism oracle only; not scientific evidence"
            and v72_positive["exact"]["root_action"] == "calibrate"
            and close(v72_positive["exact"]["value"], 2.6000000000000014)
            and close(v72_positive["map"]["exact_environment_value"], -6.0)
            and close(v72_positive["map"]["normalized_regret"], 0.09555555555555557)
            and v72_positive["structural"]["fallback_count"] == 0
        ),
        "v72_external_is_harvestable_bypass_negative": bool(
            not v72e["passed"]
            and v72_row["exact"]["root_action"] == "west"
            and close(v72_row["exact"]["value"], 13.786874999999998)
            and close(v72_row["map"]["normalized_regret"], 0.0)
            and close(v72_row["posterior_sampling"]["normalized_regret"], 0.0)
            and close(v72_row["open_loop"]["normalized_exact_advantage"], 0.0)
            and v72_row["support"]["fallback_count"] == 0
        ),
        "v73_stops_on_immaterial_economic_lower_bound": bool(
            not v73["passed"]
            and close(v73["fixed_adaptive_policy_value"], -118.941844504576)
            and close(v73["best_open_loop_value"], -119.43454720000003)
            and close(v73["fixed_adaptive_over_open_loop_normalized_advantage"], 0.0004026591008991246)
            and v73["access"]["exact_Bayes_adaptive_calls"] == 0
            and v73["access"]["MAP_calls"] == 0
            and v73["access"]["protected_source_access_count"] == 0
        ),
        "v74_is_positive_source_grounded_development": bool(
            v74["passed"]
            and close(v74_model["exact"]["value"], 5.609354999999997)
            and close(v74_model["map"]["exact_environment_value"], -44.20125)
            and close(v74_model["map"]["normalized_regret"], 0.15874625129471756)
            and close(v74_model["open_loop"]["normalized_exact_advantage"], 0.02242245239423153)
            and v74_model["integrity"]["fallback_count"] == 0
            and "not an unchanged external environment" in v74["claim_boundary"]
        ),
        "v75_is_positive_outcome_untouched_not_discovery_clean_replication": bool(
            v75["passed"]
            and close(v75_model["exact"]["value"], 0.1663984375)
            and close(v75_model["map"]["exact_environment_value"], -0.004286875000000003)
            and close(v75_model["map"]["normalized_regret"], 0.02300418646180801)
            and close(v75_model["open_loop"]["normalized_exact_advantage"], 0.022426421038444694)
            and v75_model["integrity"]["fallback_count"] == 0
            and locks["V75"]["outcome"]["source_policy_outcome_untouched"]
            and not locks["V75"]["outcome"]["source_discovery_clean"]
        ),
        "zero_human_model_and_adapter_access_after_v71": no_human_model_or_adapter,
    }


def audit_registered_artifacts() -> dict[str, bool]:
    synthesis = load_json(SYNTHESIS_CONFIG)
    census = load_json(CENSUS_CONFIG)
    synthesis_doc = (PROJECT_ROOT / SYNTHESIS_DOCUMENT).read_text()
    census_plan = (PROJECT_ROOT / CENSUS_PLAN).read_text()
    direction = (PROJECT_ROOT / RESEARCH_DIRECTION).read_text()
    synthesis_doc_normalized = " ".join(synthesis_doc.split())
    census_plan_normalized = " ".join(census_plan.split())
    direction_normalized = " ".join(direction.split())

    evidence_stages = [item["stage"] for item in synthesis["evidenceInputs"]]
    excluded_repositories = census["priorExposureExclusions"]["repositoryWide"]
    partition = census["deterministicRolePartition"]
    structural = census["postPartitionStructuralGates"]
    economic = census["prospectiveEconomicGate"]

    return {
        "synthesis_registers_complete_stage_order": evidence_stages
        == ["V71", "V72-oracle", "V72-external", "V73", "V74", "V75"],
        "synthesis_preserves_discovery_clean_boundary": bool(
            not synthesis["claimBoundary"]["discoveryCleanIndependentConfirmation"]
            and synthesis["claimBoundary"]["finiteHorizonPosteriorExpectedValueUnderLockedModels"]
            and not synthesis["sequenceClosure"]["reuseExposedSourcesForDiscoveryCleanEvidence"]
        ),
        "census_excludes_every_v62_v75_repository_family": bool(
            len(excluded_repositories) == 12
            and "https://github.com/taodav/pobax" in excluded_repositories
            and "https://github.com/cassandra/pomdp-solve" in excluded_repositories
            and "https://github.com/JuliaPOMDP/RockSample.jl" in excluded_repositories
            and "https://github.com/prateekbhustali/IMPRL" in excluded_repositories
            and "https://github.com/h2r/pomdp-py" in excluded_repositories
            and "https://github.com/kylewray/nova" in excluded_repositories
        ),
        "metadata_stage_forbids_implementation_and_outcomes": bool(
            len(census["forbiddenBeforeTheMetadataCensusLock"]) >= 7
            and not census["claimBoundary"]["thisFileSelectsAnySource"]
            and not census["claimBoundary"]["thisFileContainsAnyCandidateOutcome"]
        ),
        "roles_are_repository_disjoint_and_deterministic": bool(
            partition["minimumEligibleRepositoryDistinctFamilies"] == 2
            and partition["ordering"].startswith("ascending SHA-256")
            and partition["repositoryDisjointRoles"]
            and partition["noManualRoleSwap"]
        ),
        "later_structural_gate_forbids_new_beacon_noise_or_rewards": bool(
            structural["sourceNativeReferenceCalibrationPathway"]
            and structural["onlyPermittedProjectLayer"].startswith("a persistent uniform-prior permutation")
            and not structural["addedActionsRewardsTransitionsObservationNoiseOrBeacon"]
            and structural["commonPositiveSupportOnEveryReachablePointModelBranch"]
        ),
        "later_resource_and_economic_gates_are_fixed": bool(
            structural["maximumPhysicalStates"] == 64
            and structural["maximumActions"] == 10
            and structural["maximumObservations"] == 8
            and structural["maximumPlanningHorizonActions"] == 6
            and structural["maximumExactBellmanNodeUpperBound"] == 2_000_000
            and close(economic["minimumNormalizedAdvantage"], 0.015)
            and close(economic["minimumNormalizedMarginAboveThreshold"], 0.005)
        ),
        "documents_state_bounded_claim_and_firewall": all(
            marker in synthesis_doc_normalized
            for marker in (
                "V71–V75 form a coherent falsification sequence",
                "outcome-untouched external-domain replication",
                "discovery-clean independent confirmation has yet been completed",
                "V71–V75 are closed",
            )
        )
        and all(
            marker in census_plan_normalized
            for marker in (
                "This stage produces only a metadata inventory",
                "Manual swapping is forbidden.",
                "no action, beacon, reward, transition, or observation noise may be added",
                "confirmation implementations and all confirmation outcomes remain unopened",
            )
        ),
        "research_direction_records_v76_decision": all(
            marker in direction_normalized
            for marker in (
                "## Status after V76 (2026-08-17)",
                "V71-V75 now form a frozen falsification sequence",
                "The next authorized action is metadata-only source discovery",
                "No candidate implementation may be opened",
            )
        ),
    }


def build_lock(
    locks: dict[str, dict[str, Any]], checks: dict[str, bool], audit_sha256: str
) -> dict[str, Any]:
    source_outcomes: dict[str, Any] = {}
    for stage, relative_path in SOURCES.items():
        source = locks[stage]
        source_outcomes[stage] = {
            "outcome_lock": relative_path,
            "outcome_lock_sha256": file_sha256(PROJECT_ROOT / relative_path),
            "schema_version": source["schema_version"],
            "outcome": source["outcome"],
        }

    lock: dict[str, Any] = {
        "schema_version": "76-active-sensing-synthesis-and-source-census",
        "experiment": "v76_active_sensing_synthesis_and_source_census_lock",
        "passed": all(checks.values()),
        "checks": checks,
        "source_outcomes": source_outcomes,
        "synthesis_config": SYNTHESIS_CONFIG,
        "synthesis_config_sha256": file_sha256(PROJECT_ROOT / SYNTHESIS_CONFIG),
        "synthesis_document": SYNTHESIS_DOCUMENT,
        "synthesis_document_sha256": file_sha256(PROJECT_ROOT / SYNTHESIS_DOCUMENT),
        "source_census_config": CENSUS_CONFIG,
        "source_census_config_sha256": file_sha256(PROJECT_ROOT / CENSUS_CONFIG),
        "source_census_plan": CENSUS_PLAN,
        "source_census_plan_sha256": file_sha256(PROJECT_ROOT / CENSUS_PLAN),
        "research_direction_snapshot": RESEARCH_DIRECTION,
        "research_direction_snapshot_sha256": file_sha256(PROJECT_ROOT / RESEARCH_DIRECTION),
        "audit": AUDIT,
        "audit_sha256": audit_sha256,
        "auditor": AUDITOR,
        "auditor_sha256": file_sha256(PROJECT_ROOT / AUDITOR),
        "evidence": {
            "negative_boundary_stages": ["V71", "V72-external", "V73"],
            "mechanism_only_stage": "V72-oracle",
            "positive_development_stage": "V74",
            "positive_outcome_untouched_replication_stage": "V75",
            "discovery_clean_confirmation_stage": None,
        },
        "claim_boundary": {
            "finite_horizon_exact_joint_model_mechanism": True,
            "universal_advantage": False,
            "unchanged_external_environment": False,
            "discovery_clean_confirmation": False,
            "approximate_inference_human_model_or_real_world_claim": False,
        },
        "authorization": {
            "report_and_synthesize_V71_through_V75": True,
            "execute_metadata_only_source_census": True,
            "read_public_landing_pages_abstracts_package_and_repository_metadata": True,
            "freeze_complete_eligible_inventory_and_deterministic_role_partition": True,
            "clone_or_download_candidate_sources": False,
            "inspect_candidate_implementation_model_config_test_notebook_or_data_files": False,
            "run_candidate_simulators_parsers_exporters_or_planners": False,
            "compute_candidate_policy_values_actions_regrets_EIG_or_mutual_information": False,
            "open_protected_confirmation_implementation_or_outcomes": False,
            "modify_or_rerun_V71_through_V75": False,
            "reuse_prior_exposed_repositories_or_domain_families": False,
            "revise_V76_exposure_structural_resource_economic_or_role_gates": False,
            "human_data": False,
            "model_access": False,
            "adapter_training": False,
            "run_SMC2": False,
        },
    }
    lock["lock_payload_sha256"] = canonical_hash(lock)
    return lock


def main() -> None:
    locks = {stage: load_json(path) for stage, path in SOURCES.items()}
    checks = {**audit_outcomes(locks), **audit_registered_artifacts()}
    audit = {
        "schema_version": "76-active-sensing-synthesis-audit",
        "experiment": "v76_active_sensing_synthesis_audit",
        "passed": all(checks.values()),
        "checks": checks,
    }
    audit_path = PROJECT_ROOT / AUDIT
    write_json(audit_path, audit)
    lock = build_lock(locks, checks, file_sha256(audit_path))
    write_json(PROJECT_ROOT / LOCK, lock)
    print(json.dumps(audit, indent=2, sort_keys=True))
    if not audit["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
