#!/usr/bin/env python3
from __future__ import annotations

import json
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
        "config": PROJECT_ROOT / "configs/v214-deterministic-candidate-version-space-controls.json",
        "plan": PROJECT_ROOT / "docs/v214-deterministic-candidate-version-space-controls-plan.md",
        "roadmap": PROJECT_ROOT / "docs/research-roadmap-after-v213.md",
        "protocol": PROJECT_ROOT / "python/v214_deterministic_candidate_version_space_controls.py",
        "tests": PROJECT_ROOT / "python/test_v214_deterministic_candidate_version_space_controls.py",
        "control_worker": PROJECT_ROOT / "python/v214_control_worker.py",
        "runner": PROJECT_ROOT / "python/run_v214_deterministic_candidate_version_space_controls.py",
        "verifier": PROJECT_ROOT / "python/verify_and_freeze_v214_deterministic_candidate_version_space_controls_outcome.py",
        "auditor": PROJECT_ROOT / "python/audit_and_freeze_v214_deterministic_candidate_version_space_controls.py",
    }
    audit_path = PROJECT_ROOT / "outputs/v214-deterministic-controls/design-audit.json"
    lock_path = PROJECT_ROOT / "configs/v214-deterministic-candidate-version-space-controls-lock.json"
    output_root = PROJECT_ROOT / "outputs/v214-deterministic-controls/evaluation"
    outcome_path = PROJECT_ROOT / "configs/v214-deterministic-candidate-version-space-controls-outcome-lock.json"
    if any(path.exists() for path in (audit_path, lock_path, output_root, outcome_path)):
        raise RuntimeError("V214 is already audited, frozen, run, or outcome-frozen")
    config = json.loads(paths["config"].read_text())
    parent_path = PROJECT_ROOT / config["parentV213OutcomeLock"]
    parent = json.loads(parent_path.read_text())
    v213_lock_path = PROJECT_ROOT / parent["design_lock"]
    v213_lock = json.loads(v213_lock_path.read_text())
    input_contract = config["inputContract"]
    subsplit = config["developmentSubsplit"]
    methods = config["methods"]
    gates = config["evaluationGates"]
    eligibility = config["modelEligibilityRule"]
    worker_source = paths["control_worker"].read_text()
    exposure = config["preLockExposure"]
    checks = {
        "V213_is_frozen_positive_and_authorizes_V214_design_only": bool(
            valid_lock(parent)
            and parent["outcome"]["passed"]
            and parent["outcome"]["branch"] == "V214_DETERMINISTIC_CONTROL_DESIGN_ELIGIBLE"
            and parent["authorization"]["design_V214_deterministic_candidate_and_version_space_controls"]
            and not parent["authorization"]["run_V214_without_separate_lock"]
            and not parent["authorization"]["open_protected_downstream_or_run_model"]
        ),
        "V213_design_inputs_and_public_semantics_are_exactly_frozen": bool(
            valid_lock(v213_lock)
            and file_sha256(v213_lock_path) == parent["design_lock_sha256"]
            and file_sha256(PROJECT_ROOT / v213_lock["config"]) == v213_lock["config_sha256"]
            and file_sha256(PROJECT_ROOT / v213_lock["protocol"]) == v213_lock["protocol_sha256"]
            and file_sha256(PROJECT_ROOT / v213_lock["parent_public_semantics"]) == v213_lock["parent_public_semantics_sha256"]
        ),
        "development_only_reconstruction_contract_closes_protected_paths": bool(
            input_contract["reconstructOnlyV213DevelopmentGroupsFromFrozenGenerator"]
            and not input_contract["openV213FrozenPublicRecords"]
            and not input_contract["openV213FrozenSealedTruth"]
            and not input_contract["openV213FrozenSplit"]
            and not input_contract["constructProtectedGroupRecords"]
            and input_contract["developmentGroupCount"] == 80
            and input_contract["developmentRecordCount"] == 320
        ),
        "balanced_fit_evaluation_subsplit_is_frozen": bool(
            subsplit["fitGroupsPerFamily"] == 4
            and subsplit["evaluationGroupsPerFamily"] == 4
            and subsplit["fitGroupCount"] == 40
            and subsplit["evaluationGroupCount"] == 40
            and subsplit["fitRecordCount"] == 160
            and subsplit["evaluationRecordCount"] == 160
            and subsplit["allVariantsRemainTogether"]
        ),
        "control_ladder_budgets_ties_statuses_and_fallback_are_frozen": bool(
            len(methods["ordered"]) == 6
            and methods["ordered"][0] == "EXACT_STRUCTURAL_CEILING"
            and methods["NORMALIZED_EXACT_RETRIEVAL_K8"]["candidateBudget"] == 8
            and methods["TYPED_APPROX_RETRIEVAL_K8"]["candidateBudget"] == 8
            and methods["TYPED_APPROX_RETRIEVAL_K8"]["nearestFitGroupCount"] == 4
            and methods["BOUNDED_L0_LPLUS_SYNTHESIS"]["representedBehaviorBudget"] == 15
            and methods["FULL_CONSTRAINT_PROPAGATION"]["behaviorDomainSize"] == 256
            and methods["DETERMINISTIC_STACK"]["useTypedParsingThenBoundedSynthesisThenFullConstraintFallback"]
            and config["statusContract"]["noProposalAction"] == "DEFER_ADJUDICATE"
            and config["statusContract"]["outsideAction"] == "DEFER_OUTSIDE"
        ),
        "control_worker_has_fit_labels_but_no_evaluation_truth_interface": bool(
            "--fit-labels" in worker_source
            and "--evaluation-public" in worker_source
            and "--predictions" in worker_source
            and "evaluation-truth" not in worker_source
            and "expected_candidate_ids" not in worker_source
            and "concept_family" not in worker_source
        ),
        "noncompensatory_exactness_separation_freeze_and_eligibility_gates_are_frozen": bool(
            gates["requiredStructuralCeilingExactVersionSpaceAccuracy"] == 1.0
            and gates["requiredFullConstraintExactVersionSpaceAccuracy"] == 1.0
            and gates["requiredDeterministicStackExactVersionSpaceAccuracy"] == 1.0
            and gates["requiredDeterministicStackActionAccuracy"] == 1.0
            and gates["maximumDeterministicStackNormalizedDecisionRegret"] == 0.0
            and gates["minimumImperfectBoundedControlCount"] >= 1
            and gates["requiredGroupVariantInvariance"] == 1.0
            and eligibility["minimumResidualGroupCount"] == 8
            and eligibility["minimumAverageNormalizedDecisionRegret"] == 0.02
        ),
        "prelock_exposure_zero_and_successor_authority_narrow": bool(
            all(value == 0 for value in exposure.values())
            and config["decisionRule"]["passAuthorizesOnlySeparateSuccessorDesign"]
            and not config["decisionRule"]["passAuthorizesProtectedAccessOrModelRun"]
            and not config["decisionRule"]["passAuthorizesExternalPayloadRegistrationMutationActionOrExecution"]
        ),
        "all_required_files_exist_and_formal_outputs_absent": bool(
            all(path.is_file() for path in (*paths.values(), parent_path, v213_lock_path))
            and not output_root.exists()
            and not outcome_path.exists()
        ),
    }
    passed = all(checks.values())
    audit = {
        "schema_version": "214-deterministic-candidate-version-space-controls-design-audit",
        "experiment": config["experiment"],
        "passed": passed,
        "decision": "freeze_and_authorize_one_V214_development_control_run" if passed else "reject_V214_design",
        "checks": checks,
    }
    write_json(audit_path, audit)
    if not passed:
        print(json.dumps(audit, indent=2, sort_keys=True))
        raise SystemExit(1)
    dependencies = {
        **paths,
        "parent_V213_outcome": parent_path,
        "parent_V213_design_lock": v213_lock_path,
        "parent_V213_config": PROJECT_ROOT / v213_lock["config"],
        "parent_V213_protocol": PROJECT_ROOT / v213_lock["protocol"],
        "parent_public_semantics": PROJECT_ROOT / v213_lock["parent_public_semantics"],
        "design_audit": audit_path,
    }
    lock: dict[str, Any] = {
        "schema_version": "214-deterministic-candidate-version-space-controls-lock",
        "experiment": config["experiment"],
        "config_payload": config,
        "authorization": {
            "run_one_development_only_deterministic_control_study": True,
            "open_V213_protected_public_or_truth": False,
            "run_local_API_model_or_training": False,
            "read_external_payload_register_mutate_call_act_execute": False,
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
