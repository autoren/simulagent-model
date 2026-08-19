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
    config_path = PROJECT_ROOT / "configs/v105-open-world-interface.json"
    parent_path = PROJECT_ROOT / "configs/v104-massive-language-extraction-outcome-lock.json"
    plan_path = PROJECT_ROOT / "docs/v105-open-world-interface-plan.md"
    protocol_path = PROJECT_ROOT / "python/v105_open_world_interface.py"
    tests_path = PROJECT_ROOT / "python/test_v105_open_world_interface.py"
    runner_path = PROJECT_ROOT / "python/run_v105_open_world_interface.py"
    verifier_path = PROJECT_ROOT / "python/verify_and_freeze_v105_interface_outcome.py"
    auditor_path = PROJECT_ROOT / "python/audit_and_freeze_v105_interface.py"
    audit_path = PROJECT_ROOT / "outputs/v105-open-world-interface/interface-design-audit.json"
    lock_path = PROJECT_ROOT / "configs/v105-open-world-interface-lock.json"
    output_root = PROJECT_ROOT / "outputs/v105-open-world-interface/interface"
    if audit_path.exists() or lock_path.exists() or output_root.exists():
        raise RuntimeError("V105 interface is already frozen or materialized")
    config = json.loads(config_path.read_text())
    parent = json.loads(parent_path.read_text())
    archive_path = PROJECT_ROOT / config["sourceArchive"]
    inventory_path = PROJECT_ROOT / config["sourceInventory"]
    population_path = PROJECT_ROOT / config["selectedPopulation"]
    response = config["responseContract"]
    universe = config["safeHypothesisUniverse"]
    control = config["controlledInsufficientEvidence"]
    authority = config["authorityBoundary"]
    gates = config["interfaceGates"]
    exposure = config["preLockExposure"]
    expected_intents = {
        "calendar::calendar_remove", "calendar::calendar_set", "iot::iot_cleaning",
        "iot::iot_coffee", "iot::iot_hue_lightchange", "iot::iot_hue_lightdim",
        "iot::iot_hue_lightup", "iot::iot_wemo_off", "iot::iot_wemo_on",
        "recommendation::recommendation_events", "recommendation::recommendation_locations",
        "recommendation::recommendation_movies",
    }
    checks = {
        "V104_outcome_is_exact_and_authorizes_interface_preregistration": bool(
            valid_lock(parent) and parent["outcome"]["passed"]
            and parent["authorization"]["preregister_benchmark_interface_prompt_controls_metrics_and_gates"]
            and not parent["authorization"]["read_protected_test_before_interface_lock"]
            and not parent["authorization"]["load_model_before_interface_and_baseline_outcomes"]
        ),
        "archive_inventory_and_population_identities_are_exact": bool(
            file_sha256(archive_path) == config["sourceArchiveSha256"]
            and file_sha256(inventory_path) == config["sourceInventorySha256"]
            and file_sha256(population_path) == config["selectedPopulationSha256"]
        ),
        "visible_catalog_and_hidden_ground_truth_boundary_are_exact": bool(
            set(config["visibleScenarios"]) == {"calendar", "iot", "recommendation"}
            and set(config["visibleDeclaredIntents"]) == expected_intents
            and len(config["visibleDeclaredIntents"]) == len(expected_intents)
            and set(config["hiddenGroundTruthMustNotAppearInVisibleCatalog"])
            == {"calendar::calendar_query", "iot::iot_hue_lightoff", "email"}
        ),
        "strict_typed_response_and_safe_fallback_are_frozen": bool(
            response["requiredKeys"] == ["status", "known_intent", "novel_scenario", "confidence"]
            and response["allowedStatuses"] == ["KNOWN", "NOVEL", "UNSUPPORTED", "ABSTAIN"]
            and not response["extraKeysAllowed"]
            and response["invalidResponseFallback"] == {
                "status": "ABSTAIN", "known_intent": None,
                "novel_scenario": None, "confidence": 0.0,
            }
        ),
        "complete_nonprunable_hypothesis_universe_is_frozen": bool(
            universe["declaredKnownHypothesisCount"] == 12
            and universe["scenarioNovelHypothesisCount"] == 3
            and universe["totalHypothesisCount"] == 17
            and universe["LLMMayRankButNeverDeleteHypotheses"]
            and universe["trueHypothesisRetentionIsDeterministicallyComplete"]
        ),
        "controlled_missing_observation_is_balanced_language_free_and_shadow_only": bool(
            control["recordsPerRole"] == 64
            and control["construction"] == "same_frozen_record_identifier_with_observation_available_false_and_no_utterance_exposure"
            and not control["naturalLanguageEvidenceClaimAllowed"]
            and control["deterministicRuntimeAction"] == "ABSTAIN_AND_ASK"
            and control["modelInvocationRole"] == "shadow_only"
        ),
        "LLM_has_no_capability_belief_action_or_execution_authority": bool(
            authority["authoritativeStateIsImmutable"]
            and not authority["LLMDefinesCapabilities"]
            and not authority["LLMPrunesHypotheses"]
            and not authority["LLMUpdatesPosterior"]
            and not authority["LLMSelectsAction"]
            and not authority["LLMExecutesActionOrTool"]
            and not authority["realToolAndServiceAccess"]
        ),
        "language_model_API_training_and_effect_boundaries_remain_closed": bool(
            all(value == 0 for value in exposure.values())
            and all(gates[key] == 0 for key in (
                "maximumSelectedDevelopmentLanguageReadCount", "maximumProtectedTestLanguageReadCount",
                "maximumManualUtteranceInspectionCount", "maximumModelLoadCount",
                "maximumModelGenerationCount", "maximumLLMAPICallCount",
                "maximumAdapterTrainingRunCount", "maximumRealServiceCallCount",
                "maximumExternalSideEffectCount",
            ))
            and not config["decisionRule"]["passAuthorizesSelectedLanguageOrModelAccess"]
            and not config["decisionRule"]["passAuthorizesAPITrainingPlanningOrExecution"]
        ),
        "plan_and_locked_code_exist": all(
            path.is_file() for path in (plan_path, protocol_path, tests_path, runner_path, verifier_path, auditor_path)
        ),
    }
    passed = all(checks.values())
    audit = {
        "schema_version": "105-open-world-interface-design-audit",
        "experiment": "v105_open_world_interface_design_audit",
        "passed": passed,
        "decision": "freeze_and_authorize_language_free_interface_compilation" if passed else "reject_V105_interface",
        "checks": checks,
        "prelock_access": exposure,
    }
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    if not passed:
        print(json.dumps(audit, indent=2, sort_keys=True))
        raise SystemExit(1)
    dependencies = {
        "config": config_path, "parent_language_outcome": parent_path,
        "source_archive": archive_path, "source_inventory": inventory_path,
        "selected_population": population_path, "plan": plan_path,
        "protocol": protocol_path, "tests": tests_path, "runner": runner_path,
        "verifier": verifier_path, "auditor": auditor_path, "design_audit": audit_path,
    }
    lock: dict[str, Any] = {
        "schema_version": "105-open-world-interface-lock",
        "experiment": "v105_open_world_interface_lock",
        "config_payload": config,
        "authorization": {
            "modify_catalog_contract_control_or_gates": False,
            "read_local_archive_training_annotations_once": True,
            "read_text_free_population_identifiers_once": True,
            "read_selected_development_or_protected_test_language": False,
            "manually_inspect_any_utterance": False,
            "design_or_run_language_baseline_or_model": False,
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
