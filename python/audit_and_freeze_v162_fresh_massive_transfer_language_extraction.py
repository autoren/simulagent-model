#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
from typing import Any

from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT
from v93_open_set_source import canonical_sha256


def payload_hash(value: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def valid_lock(payload: dict[str, Any]) -> bool:
    return payload_hash(
        {key: value for key, value in payload.items() if key != "lock_payload_sha256"}
    ) == payload.get("lock_payload_sha256")


def main() -> None:
    config_path = (
        PROJECT_ROOT
        / "configs/v162-fresh-massive-transfer-language-extraction.json"
    )
    parent_path = (
        PROJECT_ROOT
        / "configs/v161-fresh-massive-transfer-population-outcome-lock.json"
    )
    plan_path = (
        PROJECT_ROOT
        / "docs/v162-fresh-massive-transfer-language-extraction-plan.md"
    )
    roadmap_path = PROJECT_ROOT / "docs/research-branches-after-v161.md"
    protocol_path = (
        PROJECT_ROOT / "python/v162_fresh_massive_transfer_language_extraction.py"
    )
    tests_path = (
        PROJECT_ROOT
        / "python/test_v162_fresh_massive_transfer_language_extraction.py"
    )
    runner_path = (
        PROJECT_ROOT
        / "python/run_v162_fresh_massive_transfer_language_extraction.py"
    )
    verifier_path = (
        PROJECT_ROOT
        / "python/verify_and_freeze_v162_fresh_massive_transfer_language_outcome.py"
    )
    auditor_path = (
        PROJECT_ROOT
        / "python/audit_and_freeze_v162_fresh_massive_transfer_language_extraction.py"
    )
    audit_path = (
        PROJECT_ROOT
        / "outputs/v162-fresh-massive-transfer-language/language-design-audit.json"
    )
    lock_path = (
        PROJECT_ROOT
        / "configs/v162-fresh-massive-transfer-language-extraction-lock.json"
    )
    output_root = (
        PROJECT_ROOT
        / "outputs/v162-fresh-massive-transfer-language/selected-language"
    )
    if audit_path.exists() or lock_path.exists() or output_root.exists():
        raise RuntimeError("V162 extraction is already frozen or materialized")

    config = json.loads(config_path.read_text())
    parent = json.loads(parent_path.read_text())
    population_path = PROJECT_ROOT / config["selectedPopulation"]
    inventory_path = PROJECT_ROOT / config["sourceInventory"]
    archive_path = PROJECT_ROOT / config["sourceArchive"]
    population = json.loads(population_path.read_text())
    gates = config["extractionGates"]
    exposure = config["preLockExposure"]

    checks = {
        "V161_population_outcome_is_exact_and_authorizes_extraction_preregistration": bool(
            valid_lock(parent)
            and parent["outcome"]["passed"]
            and parent["outcome"]["scientific_population_feasibility_passed"]
            and parent["authorization"][
                "preregister_automatic_selected_language_extraction"
            ]
            and not parent["authorization"][
                "reopen_archive_or_extract_language_before_extraction_lock"
            ]
            and parent["population"] == config["selectedPopulation"]
            and parent["population_sha256"] == config["selectedPopulationSha256"]
            and parent["selected_population_payload_sha256"]
            == config["selectedPopulationPayloadSha256"]
        ),
        "population_inventory_and_archive_identities_are_exact": bool(
            file_sha256(population_path) == config["selectedPopulationSha256"]
            and canonical_sha256(population["selected_population"])
            == config["selectedPopulationPayloadSha256"]
            and file_sha256(inventory_path) == config["sourceInventorySha256"]
            and file_sha256(archive_path) == config["sourceArchiveSha256"]
        ),
        "two_roles_and_exact_balanced_counts_are_frozen": bool(
            set(config["roles"])
            == {"development_transfer", "protected_transfer"}
            and all(
                role["expectedRecordCount"] == 192
                for role in config["roles"].values()
            )
            and config["expectedClassCountPerRole"] == 48
            and gates["requiredTotalRecordCount"] == 384
            and gates["requiredRecordCountPerRole"] == 192
            and gates["requiredRecordCountPerClassPerRole"] == 48
        ),
        "record_contract_and_exact_reconstruction_gates_are_frozen": bool(
            set(config["emittedRecordFields"])
            == {
                "record_id",
                "candidate_id",
                "source_id",
                "role",
                "source_partition",
                "class_label",
                "schema_visibility",
                "scenario",
                "intent",
                "utterance",
                "annotated_utterance",
                "slots",
                "current_utterance_intent_overlap_count",
            }
            and gates["requireExactSelectedIdentifierSet"]
            and gates["requireExactStructuralGroundTruthMatch"]
            and gates["requireExactFamiliarityReconstruction"]
            and gates["requireExactSlotTypeCountReconstruction"]
            and gates["requireDevelopmentProtectedRoleDisjointness"]
            and gates["maximumUnselectedLanguageRecordCount"] == 0
        ),
        "protected_and_model_boundaries_remain_closed": bool(
            config["roles"]["protected_transfer"]["postExtractionAccess"]
            == "sealed_until_development_policy_controls_metrics_gates_and_evaluator_are_frozen"
            and not config["decisionRule"][
                "passAuthorizesManualDevelopmentInspection"
            ]
            and not config["decisionRule"][
                "passAuthorizesManualProtectedInspection"
            ]
            and not config["decisionRule"]["passAuthorizesImmediateInterfaceScoring"]
            and not config["decisionRule"]["passAuthorizesModelInference"]
            and not config["decisionRule"][
                "passAuthorizesAPITrainingInductionPlanningActionOrExecution"
            ]
            and all(value == 0 for value in exposure.values())
            and all(
                gates[key] == 0
                for key in (
                    "maximumManualDevelopmentUtteranceInspectionCount",
                    "maximumManualProtectedUtteranceInspectionCount",
                    "maximumProtectedLanguageReadDuringDevelopmentCount",
                    "maximumModelLoadCount",
                    "maximumModelGenerationCount",
                    "maximumLLMAPICallCount",
                    "maximumAdapterTrainingRunCount",
                    "maximumRealServiceCallCount",
                    "maximumExternalSideEffectCount",
                    "maximumActualExecutionCount",
                )
            )
        ),
        "plan_roadmap_and_locked_code_exist": all(
            path.is_file()
            for path in (
                plan_path,
                roadmap_path,
                protocol_path,
                tests_path,
                runner_path,
                verifier_path,
                auditor_path,
            )
        ),
    }
    passed = all(checks.values())
    audit = {
        "schema_version": "162-fresh-massive-transfer-language-extraction-design-audit",
        "experiment": "v162_fresh_massive_transfer_language_extraction_design_audit",
        "passed": passed,
        "decision": (
            "freeze_and_authorize_exact_V161_selected_language_extraction"
            if passed
            else "reject_V162_extraction"
        ),
        "checks": checks,
        "prelock_access": exposure,
    }
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    if not passed:
        print(json.dumps(audit, indent=2, sort_keys=True))
        raise SystemExit(1)

    dependencies = {
        "config": config_path,
        "parent_population_outcome": parent_path,
        "selected_population": population_path,
        "source_inventory": inventory_path,
        "source_archive": archive_path,
        "plan": plan_path,
        "roadmap": roadmap_path,
        "protocol": protocol_path,
        "tests": tests_path,
        "runner": runner_path,
        "verifier": verifier_path,
        "auditor": auditor_path,
        "design_audit": audit_path,
    }
    lock: dict[str, Any] = {
        "schema_version": "162-fresh-massive-transfer-language-extraction-lock",
        "experiment": "v162_fresh_massive_transfer_language_extraction_lock",
        "config_payload": config,
        "authorization": {
            "modify_identifiers_fields_roles_or_gates": False,
            "read_local_archive_and_emit_exact_selected_language_once": True,
            "automatically_reconstruct_for_outcome_verification_once": True,
            "manually_inspect_development_or_protected_language": False,
            "read_protected_language_during_development": False,
            "design_or_run_interface_policy_baseline_or_model": False,
            "load_local_or_API_model": False,
            "train_adapter_or_learn_likelihood": False,
            "grant_model_capability_state_belief_action_or_execution_authority": False,
            "perform_real_service_call_external_side_effect_or_execution": False,
        },
    }
    for key, path in dependencies.items():
        lock[key] = str(path.relative_to(PROJECT_ROOT))
        lock[f"{key}_sha256"] = file_sha256(path)
    lock["lock_payload_sha256"] = payload_hash(lock)
    lock_path.write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n")
    print(json.dumps(audit, indent=2, sort_keys=True))
    print(
        json.dumps(
            {
                "lock": str(lock_path.relative_to(PROJECT_ROOT)),
                "sha256": file_sha256(lock_path),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
