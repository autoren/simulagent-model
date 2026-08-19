#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
from typing import Any

from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT


def payload_hash(value: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def valid_lock(payload: dict[str, Any]) -> bool:
    return payload_hash({key: value for key, value in payload.items() if key != "lock_payload_sha256"}) == payload.get("lock_payload_sha256")


def main() -> None:
    config_path = PROJECT_ROOT / "configs/v104-massive-language-extraction.json"
    parent_path = PROJECT_ROOT / "configs/v101-massive-population-outcome-lock.json"
    presto_path = PROJECT_ROOT / "configs/v103-presto-target-syntax-census-outcome-lock.json"
    plan_path = PROJECT_ROOT / "docs/v104-massive-language-extraction-plan.md"
    protocol_path = PROJECT_ROOT / "python/v104_massive_language_extraction.py"
    tests_path = PROJECT_ROOT / "python/test_v104_massive_language_extraction.py"
    runner_path = PROJECT_ROOT / "python/run_v104_massive_language_extraction.py"
    verifier_path = PROJECT_ROOT / "python/verify_and_freeze_v104_language_outcome.py"
    auditor_path = PROJECT_ROOT / "python/audit_and_freeze_v104_language_extraction.py"
    audit_path = PROJECT_ROOT / "outputs/v104-massive-language/language-design-audit.json"
    lock_path = PROJECT_ROOT / "configs/v104-massive-language-extraction-lock.json"
    output_root = PROJECT_ROOT / "outputs/v104-massive-language/selected-language"
    if audit_path.exists() or lock_path.exists() or output_root.exists():
        raise RuntimeError("V104 extraction is already frozen or materialized")
    config = json.loads(config_path.read_text())
    parent = json.loads(parent_path.read_text())
    presto = json.loads(presto_path.read_text())
    population_path = PROJECT_ROOT / config["selectedPopulation"]
    inventory_path = PROJECT_ROOT / config["sourceInventory"]
    archive_path = PROJECT_ROOT / config["sourceArchive"]
    gates = config["extractionGates"]
    exposure = config["preLockExposure"]
    checks = {
        "V101_population_outcome_is_exact_and_authorizes_extraction_preregistration": bool(
            valid_lock(parent) and parent["outcome"]["passed"]
            and parent["outcome"]["scientific_population_feasibility_passed"]
            and parent["authorization"]["preregister_selected_language_extraction"]
            and not parent["authorization"]["reopen_archive_or_extract_language_before_extraction_lock"]
        ),
        "V103_closes_PRESTO_and_does_not_authorize_population": bool(
            valid_lock(presto) and presto["outcome"]["passed"]
            and not presto["outcome"]["diagnostic_viability_passed"]
            and presto["authorization"]["close_PRESTO_paired_insufficiency_branch"]
            and not presto["authorization"]["preregister_new_PRESTO_dependency_construction"]
        ),
        "population_inventory_and_archive_identities_are_exact": bool(
            file_sha256(population_path) == config["selectedPopulationSha256"]
            and file_sha256(inventory_path) == config["sourceInventorySha256"]
            and file_sha256(archive_path) == config["sourceArchiveSha256"]
        ),
        "two_roles_and_exact_balanced_counts_are_frozen": bool(
            set(config["roles"]) == {"development", "protected_test"}
            and all(role["expectedRecordCount"] == 256 for role in config["roles"].values())
            and config["expectedClassCountPerRole"] == 64
            and gates["requiredTotalRecordCount"] == 512
            and gates["requiredRecordCountPerRole"] == 256
            and gates["requiredRecordCountPerClassPerRole"] == 64
        ),
        "record_contract_and_exact_reconstruction_gates_are_frozen": bool(
            "utterance" in config["emittedRecordFields"]
            and "annotated_utterance" in config["emittedRecordFields"]
            and "slots" in config["emittedRecordFields"]
            and gates["requireExactSelectedIdentifierSet"]
            and gates["requireExactStructuralGroundTruthMatch"]
            and gates["requireExactFamiliarityReconstruction"]
            and gates["requireExactSlotTypeCountReconstruction"]
            and gates["requireDevelopmentProtectedTestDisjointness"]
            and gates["maximumUnselectedLanguageRecordCount"] == 0
        ),
        "protected_test_and_model_boundaries_remain_closed": bool(
            config["roles"]["protected_test"]["postExtractionAccess"]
            == "sealed_until_prompt_controls_metrics_and_gates_lock"
            and not config["decisionRule"]["passAuthorizesManualProtectedTestInspection"]
            and not config["decisionRule"]["passAuthorizesModelInference"]
            and not config["decisionRule"]["passAuthorizesAPITrainingPosteriorPlanningOrExecution"]
            and all(value == 0 for value in exposure.values())
            and all(gates[key] == 0 for key in (
                "maximumManualDevelopmentUtteranceInspectionCount",
                "maximumManualProtectedTestUtteranceInspectionCount", "maximumModelLoadCount",
                "maximumModelGenerationCount", "maximumLLMAPICallCount",
                "maximumAdapterTrainingRunCount", "maximumRealServiceCallCount",
                "maximumExternalSideEffectCount",
            ))
        ),
        "plan_and_locked_code_exist": all(
            path.is_file() for path in (plan_path, protocol_path, tests_path, runner_path, verifier_path, auditor_path)
        ),
    }
    passed = all(checks.values())
    audit = {
        "schema_version": "104-massive-language-extraction-design-audit",
        "experiment": "v104_massive_language_extraction_design_audit",
        "passed": passed,
        "decision": "freeze_and_authorize_exact_selected_language_extraction" if passed else "reject_V104_extraction",
        "checks": checks, "prelock_access": exposure,
    }
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    if not passed:
        print(json.dumps(audit, indent=2, sort_keys=True))
        raise SystemExit(1)
    dependencies = {
        "config": config_path, "parent_population_outcome": parent_path,
        "presto_closure_outcome": presto_path, "selected_population": population_path,
        "source_inventory": inventory_path, "source_archive": archive_path,
        "plan": plan_path, "protocol": protocol_path, "tests": tests_path,
        "runner": runner_path, "verifier": verifier_path, "auditor": auditor_path,
        "design_audit": audit_path,
    }
    lock: dict[str, Any] = {
        "schema_version": "104-massive-language-extraction-lock",
        "experiment": "v104_massive_language_extraction_lock",
        "config_payload": config,
        "authorization": {
            "modify_identifiers_fields_roles_or_gates": False,
            "read_local_archive_and_emit_exact_selected_language_once": True,
            "manually_inspect_development_or_protected_test_language": False,
            "design_or_run_prompt_baseline_or_model": False,
            "load_local_or_API_model": False,
            "train_adapter_or_learn_likelihood": False,
            "grant_model_capability_state_belief_action_or_execution_authority": False,
            "perform_real_service_call_or_external_side_effect": False,
        },
    }
    for key, path in dependencies.items():
        lock[key] = str(path.relative_to(PROJECT_ROOT))
        lock[f"{key}_sha256"] = file_sha256(path)
    lock["lock_payload_sha256"] = payload_hash(lock)
    lock_path.write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n")
    print(json.dumps(audit, indent=2, sort_keys=True))
    print(json.dumps({"lock": str(lock_path.relative_to(PROJECT_ROOT)), "sha256": file_sha256(lock_path)}, indent=2))


if __name__ == "__main__":
    main()
