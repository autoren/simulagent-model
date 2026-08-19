#!/usr/bin/env python3
"""Audit and freeze the V88 external-language local candidate design."""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from typing import Any

from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT


def payload_hash(value: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def main() -> None:
    design_path = PROJECT_ROOT / "configs/v88-external-intent-candidate-design.json"
    parent_path = PROJECT_ROOT / "configs/v87-external-language-source-outcome-lock.json"
    plan_path = PROJECT_ROOT / "docs/v88-external-intent-candidate-plan.md"
    auditor_path = PROJECT_ROOT / "python/audit_and_freeze_v88_external_candidate_design.py"
    audit_path = PROJECT_ROOT / "outputs/v88-external-intent-candidate/design-audit.json"
    lock_path = PROJECT_ROOT / "configs/v88-external-intent-candidate-design-lock.json"
    if audit_path.exists() or lock_path.exists():
        raise RuntimeError("V88 design is already frozen")
    if (PROJECT_ROOT / "outputs/v88-external-intent-candidate/corpus").exists():
        raise RuntimeError("V88 corpus exists before design lock")

    config = json.loads(design_path.read_text())
    parent = json.loads(parent_path.read_text())
    parent_payload = {key: value for key, value in parent.items() if key != "lock_payload_sha256"}
    inventory = json.loads((PROJECT_ROOT / parent["inventory"]).read_text())
    strata = config["population"]["strata"]
    requested = Counter((item["service"], item["activeIntent"]) for item in strata for _ in range(item["count"]))
    available = Counter((row["service"], row["active_intent"]) for row in inventory["record_index"])
    stage = config["stageAuthorization"]
    gates = config["gates"]
    checks = {
        "positive_V87_parent_exact_and_authorizes_sealed_subset": bool(
            payload_hash(parent_payload) == parent["lock_payload_sha256"]
            and parent["outcome"]["passed"]
            and parent["authorization"]["preregister_sealed_nonexecutable_external_language_shadow_subset"]
            and parent["authorization"]["select_subset_before_utterance_extraction_by_frozen_hash_rule"]
            and not parent["authorization"]["access_local_or_API_model_before_subset_and_prompt_lock"]
        ),
        "source_revision_inventory_identity_and_license_exact": bool(
            config["source"]["revision"] == inventory["provenance"]["revision"]
            and config["source"]["license"] == inventory["provenance"]["license"]
            and config["source"]["inventorySha256"] == inventory["record_index_sha256"]
            and file_sha256(PROJECT_ROOT / parent["inventory"]) == parent["inventory_sha256"]
        ),
        "all_fixed_strata_are_available_before_language_extraction": all(requested[key] <= available[key] for key in requested),
        "population_is_exactly_48_with_balanced_active_and_NONE_roles": bool(
            config["population"]["recordCount"] == sum(item["count"] for item in strata) == 48
            and sum(item["count"] for item in strata if item["activeIntent"] == "NONE") == 24
            and sum(item["count"] for item in strata if item["activeIntent"] != "NONE") == 24
            and config["population"]["selectionBeforeUtteranceExtraction"]
            and config["population"]["selectionSalt"] == "simulagent-v88-external-shadow-v1"
        ),
        "prompt_and_output_contract_are_frozen_non_authoritative_and_open_set": bool(
            "always include NONE" in config["systemPrompt"]
            and config["outputContract"]["exactKeys"] == ["intent_candidates", "state_slot_key_candidates"]
            and config["outputContract"]["permanentlyNonDeployable"]
            and config["outputContract"]["mayNotEmitValuesActionsToolCallsOrFreeText"]
            and config["inputContract"]["sourceLanguageMayNotBeSentToAPI"]
            and config["inputContract"]["sourceLanguageMayNotBePrintedByBuilderOrRunner"]
        ),
        "local_model_revision_and_one_shot_decoding_are_exact": bool(
            config["model"]["provider"] == "local_mlx"
            and config["model"]["revision"] == "0e7ffd5c629ef7719d4cbc04069232580bfa9d9c"
            and config["model"]["frozen"]
            and not config["model"]["APIRequired"]
            and config["decoding"]["temperature"] == 0.0
            and config["decoding"]["samplesPerRecord"] == 1
            and not config["decoding"]["retryOnMalformedOutput"]
        ),
        "noncompensatory_quality_safety_and_access_gates_are_present": bool(
            gates["requiredRecordCount"] == 48
            and gates["requiredNoneRecordCount"] == 24
            and gates["minimumExactJSONParseRate"] == 1.0
            and gates["minimumOntologyConformanceRate"] == 1.0
            and gates["minimumMandatoryNoneInclusionRate"] == 1.0
            and gates["minimumPermanentNonDeployableRate"] == 1.0
            and gates["minimumGoldActiveIntentCoverageRate"] >= 0.9
            and gates["minimumIntentCandidateSetExactRate"] >= 0.7
            and gates["minimumStateSlotKeyRecallRate"] >= 0.75
            and gates["minimumIntentExactImprovementOverExhaustive"] >= 0.2
            and gates["maximumModelLoadCount"] == 1
            and gates["maximumModelGenerationCount"] == 48
            and all(gates[key] == 0 for key in (
                "maximumLLMAPICallCount", "maximumAdapterTrainingRunCount",
                "maximumManualUtteranceInspectionCount", "maximumRealServiceCallCount",
                "maximumExternalSideEffectCount"
            ))
        ),
        "design_stage_has_no_corpus_model_training_execution_or_manual_language_authority": bool(
            stage["auditAndFreezeDesign"]
            and not stage["selectAndSealCorpus"]
            and not stage["extractSelectedUtterancesByCode"]
            and not stage["inspectUtterancesManually"]
            and not stage["implementAndAuditRunner"]
            and not stage["runLocalModel"]
            and not stage["runAPIModel"]
            and not stage["trainAdapter"]
            and not stage["performRealServiceCall"]
            and not stage["performExternalSideEffect"]
        ),
    }
    passed = all(checks.values())
    audit = {
        "schema_version": "88-external-intent-candidate-design-audit",
        "experiment": "v88_external_intent_candidate_design_audit",
        "passed": passed,
        "decision": "freeze_design_and_authorize_hash_selected_corpus_seal" if passed else "reject_V88_design",
        "checks": checks,
        "requested_strata": {f"{key[0]}::{key[1]}": count for key, count in sorted(requested.items())},
        "available_strata": {f"{key[0]}::{key[1]}": available[key] for key in sorted(requested)},
        "access": {
            "utterance_extraction_count": 0,
            "manual_utterance_inspection_count": 0,
            "model_load_count": 0,
            "model_generation_count": 0,
            "LLM_API_call_count": 0,
            "adapter_training_run_count": 0,
            "real_service_call_count": 0,
            "external_side_effect_count": 0,
        },
    }
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    if not passed:
        print(json.dumps(audit, indent=2, sort_keys=True))
        raise SystemExit(1)

    lock = {
        "schema_version": "88-external-intent-candidate-design-lock",
        "experiment": "v88_external_intent_candidate_design_lock",
        "design": str(design_path.relative_to(PROJECT_ROOT)),
        "design_sha256": file_sha256(design_path),
        "config_payload": config,
        "parent_V87_outcome_lock": str(parent_path.relative_to(PROJECT_ROOT)),
        "parent_V87_outcome_lock_sha256": file_sha256(parent_path),
        "source_inventory": parent["inventory"],
        "source_inventory_sha256": parent["inventory_sha256"],
        "plan": str(plan_path.relative_to(PROJECT_ROOT)),
        "plan_sha256": file_sha256(plan_path),
        "design_auditor": str(auditor_path.relative_to(PROJECT_ROOT)),
        "design_auditor_sha256": file_sha256(auditor_path),
        "design_audit": str(audit_path.relative_to(PROJECT_ROOT)),
        "design_audit_sha256": file_sha256(audit_path),
        "authorization": {
            "modify_population_prompt_model_decoding_controls_or_gates": False,
            "select_extract_and_seal_corpus_once_by_frozen_code": True,
            "manually_inspect_selected_language": False,
            "implement_and_audit_local_runner_after_corpus_seal": True,
            "run_local_model_before_implementation_lock": False,
            "run_API_model": False,
            "train_adapter": False,
            "perform_real_service_call_or_external_side_effect": False,
        },
    }
    lock["lock_payload_sha256"] = payload_hash(lock)
    lock_path.write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n")
    print(json.dumps(audit, indent=2, sort_keys=True))
    print(json.dumps({"lock": str(lock_path), "sha256": file_sha256(lock_path)}, indent=2))


if __name__ == "__main__":
    main()
