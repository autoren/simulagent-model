#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from audit_and_freeze_v165_factored_ontology_identifiability_population import valid_lock
from audit_and_freeze_v167_exact_evidence_gathering_planner import payload_hash
from v10_protocol import file_sha256
from v221_deterministic_mondo_residual import derive_role_manifest
from v22r2_grounding import PROJECT_ROOT


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def main() -> None:
    paths = {
        "config": PROJECT_ROOT / "configs/v221-deterministic-mondo-residual.json",
        "plan": PROJECT_ROOT / "docs/v221-deterministic-mondo-residual-plan.md",
        "roadmap": PROJECT_ROOT / "docs/research-roadmap-after-v220.md",
        "protocol": PROJECT_ROOT / "python/v221_deterministic_mondo_residual.py",
        "tests": PROJECT_ROOT / "python/test_v221_deterministic_mondo_residual.py",
        "runner": PROJECT_ROOT / "python/run_v221_deterministic_mondo_residual.py",
        "verifier": PROJECT_ROOT / "python/verify_and_freeze_v221_deterministic_mondo_residual_outcome.py",
        "auditor": PROJECT_ROOT / "python/audit_and_freeze_v221_deterministic_mondo_residual.py",
        "inherited_parser": PROJECT_ROOT / "python/v218_mondo_artifact_population.py",
    }
    output_root = PROJECT_ROOT / "outputs/v221-deterministic-mondo-residual"
    audit_path = output_root / "design-audit.json"
    role_path = output_root / "design/role-manifest.json"
    lock_path = PROJECT_ROOT / "configs/v221-deterministic-mondo-residual-lock.json"
    outcome_path = PROJECT_ROOT / "configs/v221-deterministic-mondo-residual-outcome-lock.json"
    if output_root.exists() or lock_path.exists() or outcome_path.exists():
        raise RuntimeError("V221 is already audited, frozen, run, or outcome-frozen")
    config = json.loads(paths["config"].read_text())
    inputs = config["inputContract"]
    parent_path = PROJECT_ROOT / config["parentV220OutcomeLock"]
    parent = json.loads(parent_path.read_text())
    manifest_path = PROJECT_ROOT / inputs["populationManifest"]
    population_manifest = json.loads(manifest_path.read_text())
    role_manifest = derive_role_manifest(population_manifest["development_group_ids"], config)
    methods = [row["methodId"] for row in config["methodPortfolio"]]
    gates = config["evaluationGates"]
    residual = config["residualDefinition"]
    parent_raw = parent["raw_payload_hashes"]
    checks = {
        "V220_verified_positive_authorizes_only_development_deterministic_design": bool(
            valid_lock(parent)
            and parent["outcome"]["verification_passed"]
            and parent["outcome"]["scientific_passed"]
            and parent["outcome"]["branch"] == "FRESH_MONDO_REPRESENTATIONAL_POPULATION_ELIGIBLE"
            and parent["authorization"]["design_V221_development_only_deterministic_controls"]
            and not parent["authorization"]["open_protected_or_run_model"]
        ),
        "development_manifest_raw_and_protected_hash_contract_equals_V220": bool(
            inputs["developmentPublicSha256"] == parent["development_public_sha256"]
            and inputs["developmentTruthSha256"] == parent["development_truth_sha256"]
            and inputs["populationManifestSha256"] == parent["population_manifest_sha256"] == file_sha256(manifest_path)
            and inputs["olderOBOSha256"] == parent_raw["MONDO_BASE_2026_05_05"]
            and inputs["newerOBOSha256"] == parent_raw["MONDO_BASE_2026_06_02"]
            and inputs["protectedPublicSha256"] == parent["protected_public_sha256"]
            and inputs["protectedTruthSha256"] == parent["protected_truth_sha256"]
            and inputs["protectedJSONLMayBeHashedButNotLoaded"]
        ),
        "role_manifest_is_hash_derived_complete_disjoint_and_frozen_before_JSONL_access": bool(
            len(population_manifest["development_group_ids"]) == inputs["expectedDevelopmentGroupCount"] == 1621
            and len(population_manifest["protected_group_ids"]) == inputs["expectedProtectedGroupCount"] == 540
            and len(role_manifest["calibration_group_ids"]) == config["roleSplit"]["expectedCalibrationGroupCount"] == 1296
            and len(role_manifest["evaluation_group_ids"]) == config["roleSplit"]["expectedEvaluationGroupCount"] == 325
            and role_manifest["group_overlap_count"] == 0
            and role_manifest["source_group_accounting_exact"]
            and config["roleSplit"]["evaluationRecordsMayNotTuneMethodsBudgetsThresholdsOrController"]
        ),
        "fixed_nontrained_portfolio_catalog_semantics_and_budgets_are_complete": bool(
            methods == ["M0_NORMALIZED_EXACT", "M1_EXACT_FAMILY", "M2_HYBRID_RETRIEVAL", "M3_FINAL_FAIL_CLOSED"]
            and methods == config["metrics"]["requiredByMethod"]
            and config["candidateBudgets"] == [1, 4, 8, 16]
            and config["primaryAcceptedBudget"] == residual["budget"] == 8
            and config["catalogDesign"]["familyDefinition"].startswith("union_stable_term_id")
            and config["catalogDesign"]["equivalenceCollapse"] == "exact_state_class_identity"
            and not config["catalogDesign"]["remoteImportResolution"]
            and config["retrievalScoring"]["familyExpansionIsAtomic"]
            and config["retrievalScoring"]["partialFamilyInsertionForbidden"]
        ),
        "fail_closed_controller_primary_safety_and_metric_cells_are_frozen": bool(
            config["controller"]["versionUnspecifiedDecision"] == "PRESERVE_VERSION_SPACE_OR_CLARIFY"
            and config["controller"]["retrievalOnlyDecision"] == "PRESERVE_VERSION_SPACE_OR_CLARIFY"
            and config["controller"]["overflowDecision"] == "PRESERVE_VERSION_SPACE_OR_CLARIFY"
            and config["controller"]["conflictingLifecycleDecision"] == "PRESERVE_VERSION_SPACE_OR_CLARIFY"
            and gates["requiredCandidateClassValidity"] == 1.0
            and gates["requiredCandidateBudgetCompliance"] == 1.0
            and gates["requiredAtomicFamilyExpansionAccuracy"] == 1.0
            and gates["requiredContradictionFailClosedAccuracy"] == 1.0
            and gates["maximumUnsafeSingletonCollapseRate"] == 0.0
            and gates["maximumPrimaryEvaluationMeanDecisionRegret"] == 0.1
            and gates["requiredMetricCellCoverage"] == 1.0
            and gates["requiredNoEvaluationTuning"]
        ),
        "twelve_group_decision_relevant_residual_rule_and_branch_authority_are_frozen": bool(
            residual["methodId"] == "M3_FINAL_FAIL_CLOSED"
            and residual["minimumModelEligibleResidualGroupCount"] == 12
            and residual["requiresZeroUnsafeSingletonCollapse"]
            and residual["requiresFailClosedConflictHandlingAccuracy"] == 1.0
            and config["decisionRule"]["residualPassAuthorizesLocalModelDesignOnly"]
            and not config["decisionRule"]["passAuthorizesProtectedAccessOrModelRun"]
            and not config["decisionRule"]["passAuthorizesRegistrationMutationServiceActionOrExecution"]
        ),
        "all_inputs_and_sources_exist_but_no_development_body_or_output_was_opened": bool(
            all(path.is_file() for path in (*paths.values(), parent_path, manifest_path))
            and all((PROJECT_ROOT / inputs[key]).is_file() for key in (
                "developmentPublic", "developmentTruth", "olderOBO", "newerOBO", "protectedPublic", "protectedTruth"
            ))
            and not output_root.exists()
            and not lock_path.exists()
            and not outcome_path.exists()
        ),
    }
    passed = all(checks.values())
    audit = {
        "schema_version": "221-deterministic-mondo-residual-design-audit",
        "experiment": config["experiment"], "passed": passed,
        "decision": "freeze_roles_methods_budgets_controller_residual_and_authorize_one_development_evaluation" if passed else "reject_V221_design",
        "checks": checks,
        "development_JSONL_body_read_count_before_lock": 0,
        "protected_JSONL_body_load_count": 0,
    }
    write_json(audit_path, audit)
    write_json(role_path, role_manifest)
    if not passed:
        print(json.dumps(audit, indent=2, sort_keys=True))
        raise SystemExit(1)
    dependencies = {
        **paths, "parent_V220_outcome": parent_path, "population_manifest": manifest_path,
        "older_OBO": PROJECT_ROOT / inputs["olderOBO"], "newer_OBO": PROJECT_ROOT / inputs["newerOBO"],
        "design_audit": audit_path, "role_manifest": role_path,
    }
    lock: dict[str, Any] = {
        "schema_version": "221-deterministic-mondo-residual-lock",
        "experiment": config["experiment"], "config_payload": config,
        "authorization": {
            "load_V220_development_public_and_truth_once": True,
            "run_one_development_only_deterministic_evaluation": True,
            "hash_but_do_not_load_V220_protected": True,
            "run_model_or_open_protected": False,
            "register_mutate_service_act_execute": False,
        },
    }
    for key, path in dependencies.items():
        lock[key] = str(path.relative_to(PROJECT_ROOT))
        lock[f"{key}_sha256"] = file_sha256(path)
    # These JSONL hashes are inherited without opening their record bodies during design audit.
    for key, path_key, hash_key in (
        ("development_public", "developmentPublic", "developmentPublicSha256"),
        ("development_truth", "developmentTruth", "developmentTruthSha256"),
        ("protected_public", "protectedPublic", "protectedPublicSha256"),
        ("protected_truth", "protectedTruth", "protectedTruthSha256"),
    ):
        lock[key] = inputs[path_key]
        lock[f"{key}_sha256"] = inputs[hash_key]
    lock["lock_payload_sha256"] = payload_hash(lock)
    write_json(lock_path, lock)
    print(json.dumps(audit, indent=2, sort_keys=True))
    print(json.dumps({"lock": str(lock_path.relative_to(PROJECT_ROOT)), "sha256": file_sha256(lock_path)}, indent=2))


if __name__ == "__main__":
    main()
