#!/usr/bin/env python3
from __future__ import annotations

import json
from typing import Any

from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT
from audit_and_freeze_v165_factored_ontology_identifiability_population import valid_lock
from audit_and_freeze_v167_exact_evidence_gathering_planner import payload_hash


def write_json(path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def main() -> None:
    config_path = PROJECT_ROOT / "configs/v190-protected-multiway-menu-compression-confirmation.json"
    plan_path = PROJECT_ROOT / "docs/v190-protected-multiway-menu-compression-confirmation-plan.md"
    protocol_path = PROJECT_ROOT / "python/v190_protected_multiway_menu_compression_confirmation.py"
    tests_path = PROJECT_ROOT / "python/test_v190_protected_multiway_menu_compression_confirmation.py"
    runner_path = PROJECT_ROOT / "python/run_v190_protected_multiway_menu_compression_confirmation.py"
    verifier_path = PROJECT_ROOT / "python/verify_and_freeze_v190_protected_multiway_menu_compression_confirmation_outcome.py"
    auditor_path = PROJECT_ROOT / "python/audit_and_freeze_v190_protected_multiway_menu_compression_confirmation.py"
    audit_path = PROJECT_ROOT / "outputs/v190-protected-multiway-menu-compression-confirmation/design-audit.json"
    lock_path = PROJECT_ROOT / "configs/v190-protected-multiway-menu-compression-confirmation-lock.json"
    output_root = PROJECT_ROOT / "outputs/v190-protected-multiway-menu-compression-confirmation/confirmation"
    outcome_path = PROJECT_ROOT / "configs/v190-protected-multiway-menu-compression-confirmation-outcome-lock.json"
    if any(path.exists() for path in (audit_path, lock_path, output_root, outcome_path)):
        raise RuntimeError("V190 is already preregistered, evaluated, or frozen")
    config = json.loads(config_path.read_text())
    parent_path = PROJECT_ROOT / config["parentV189OutcomeLock"]
    parent = json.loads(parent_path.read_text())
    v189_lock = json.loads((PROJECT_ROOT / parent["feasibility_lock"]).read_text())
    v186_outcome = json.loads((PROJECT_ROOT / v189_lock["source_V186_outcome"]).read_text())
    sources = {
        "contract_catalog": PROJECT_ROOT / config["contractCatalog"],
        "protected_bindings": PROJECT_ROOT / config["protectedBindings"],
        "source_V189_result": PROJECT_ROOT / config["sourceV189Result"],
    }
    source_result = json.loads(sources["source_V189_result"].read_text())
    fixed = config["fixedPolicy"]
    gates = config["confirmationGates"]
    decision = config["decisionRule"]
    checks = {
        "V189_is_valid_formal_failure_with_frozen_exploratory_pattern": bool(
            valid_lock(parent)
            and parent["outcome"]["passed"]
            and not parent["outcome"]["scientific_feasibility_gates_passed"]
            and not source_result["passed"]
            and source_result["summary"]["robust_multiway_value"]
            and source_result["summary"]["pure_bit_slot_exact_cost"] < 0.40
        ),
        "source_contract_and_protected_binding_artifacts_are_exact": bool(
            file_sha256(sources["contract_catalog"]) == v189_lock["contract_catalog_sha256"]
            and file_sha256(sources["protected_bindings"]) == v186_outcome["protected_bindings_sha256"]
            and file_sha256(sources["source_V189_result"]) == parent["result_sha256"]
        ),
        "policy_and_pricing_are_fixed_without_protected_optimization": bool(
            fixed["questionSequence"] == ["M189_domain", "M189_intent_concept"]
            and fixed["singletonEarlyStopping"]
            and fixed["questionSelectionUsesNoProtectedPriorTargetOrOutcome"]
            and fixed["noPolicyOptimizationOrReselection"]
            and fixed["genericTrustedClarificationCost"] == 0.40
        ),
        "fresh_confirmation_gates_are_coherent_and_noncompensatory": bool(
            gates["requiredProtectedBindingCount"] == 132
            and gates["requiredObservedProtectedCount"] == 120
            and gates["requiredMissingProtectedCount"] == 12
            and gates["requiredObservedFinalExactnessRate"] == 1.0
            and gates["minimumImprovementOverAlwaysGeneric"] > 0
            and gates["requiredPolicyOptimizationCount"] == 0
            and gates["maximumProtectedUtteranceLanguageReadCount"] == 0
        ),
        "successor_authority_and_prelock_exposure_are_closed": bool(
            not decision["passAuthorizesImmediateLanguageModelUIOrHumanClaim"]
            and not decision["passAuthorizesProtectedUtteranceAccess"]
            and not decision["passAuthorizesRegistrationAuthorityActionOrExecution"]
            and all(value == 0 for value in config["preLockExposure"].values())
        ),
        "required_files_exist_and_outputs_are_absent": bool(
            all(path.is_file() for path in (
                config_path, parent_path, plan_path, protocol_path, tests_path,
                runner_path, verifier_path, auditor_path, *sources.values(),
            )) and not output_root.exists()
        ),
    }
    passed = all(checks.values())
    audit = {
        "schema_version": "190-protected-multiway-menu-compression-confirmation-design-audit",
        "experiment": config["experiment"],
        "passed": passed,
        "decision": "freeze_and_authorize_one_fixed_V190_protected_identity_confirmation" if passed else "reject_V190_design",
        "checks": checks,
        "prelock_exposure": config["preLockExposure"],
        "protected_target_path_score_count": 0,
        "policy_optimization_count": 0,
        "protected_utterance_language_read_count": 0,
        "model_load_count": 0,
        "actual_execution_count": 0,
    }
    write_json(audit_path, audit)
    if not passed:
        print(json.dumps(audit, indent=2, sort_keys=True))
        raise SystemExit(1)
    dependencies = {
        "config": config_path, "parent_V189_outcome": parent_path,
        "source_V189_lock": PROJECT_ROOT / parent["feasibility_lock"],
        **sources, "plan": plan_path, "protocol": protocol_path, "tests": tests_path,
        "runner": runner_path, "verifier": verifier_path, "auditor": auditor_path,
        "design_audit": audit_path,
    }
    lock: dict[str, Any] = {
        "schema_version": "190-protected-multiway-menu-compression-confirmation-lock",
        "experiment": config["experiment"],
        "config_payload": config,
        "authorization": {
            "modify_policy_pricing_controls_gates_or_decision": False,
            "run_fixed_protected_identity_confirmation_once": True,
            "optimize_policy_or_read_protected_utterance_language": False,
            "run_model_API_training_UI_or_human_study": False,
            "register_mutate_call_service_act_or_execute": False,
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
