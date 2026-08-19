#!/usr/bin/env python3
"""Audit and freeze the V90 multi-model comparison before language extraction or weights."""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from typing import Any

from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT


def payload_hash(value: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def selection_hash(salt: str, service: str, intent: str, record_id: str) -> str:
    return hashlib.sha256(f"{salt}\0{service}\0{intent}\0{record_id}".encode()).hexdigest()


def structurally_select(config: dict[str, Any], inventory: dict[str, Any]) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    used_dialogues: set[str] = set()
    salt = config["population"]["selectionSalt"]
    for stratum in config["population"]["strata"]:
        candidates = [
            row for row in inventory["record_index"]
            if row["service"] == stratum["service"] and row["active_intent"] == stratum["activeIntent"]
        ]
        candidates.sort(key=lambda row: (
            selection_hash(salt, stratum["service"], stratum["activeIntent"], row["record_id"]),
            row["record_id"],
        ))
        chosen = []
        for row in candidates:
            if row["dialogue_id"] in used_dialogues:
                continue
            chosen.append(row)
            used_dialogues.add(row["dialogue_id"])
            if len(chosen) == stratum["count"]:
                break
        if len(chosen) != stratum["count"]:
            raise RuntimeError(f"V90 structurally infeasible unique-dialogue stratum: {stratum}")
        selected.extend(chosen)
    return selected


def main() -> None:
    design_path = PROJECT_ROOT / "configs/v90-capacity-generation-design.json"
    source_path = PROJECT_ROOT / "configs/v90-capacity-generation-source-outcome-lock.json"
    plan_path = PROJECT_ROOT / "docs/v90-capacity-generation-plan.md"
    auditor_path = PROJECT_ROOT / "python/audit_and_freeze_v90_design.py"
    builder_path = PROJECT_ROOT / "python/build_v90_capacity_corpus.py"
    audit_path = PROJECT_ROOT / "outputs/v90-capacity-generation/design-audit.json"
    lock_path = PROJECT_ROOT / "configs/v90-capacity-generation-design-lock.json"
    corpus_root = PROJECT_ROOT / "data/v90-capacity-generation"
    prior_audit_path = audit_path.with_name("design-audit-attempt-0.json")
    if lock_path.exists() or corpus_root.exists() or prior_audit_path.exists():
        raise RuntimeError("V90 experiment design is already frozen, materialized, or mechanically retried")
    prior_auditor_attempt = None
    if audit_path.exists():
        prior_auditor_attempt = json.loads(audit_path.read_text())
        if prior_auditor_attempt.get("decision") != "reject_V90_model_comparison_design":
            raise RuntimeError("unexpected prior V90 design-audit artifact")
        audit_path.replace(prior_audit_path)

    config = json.loads(design_path.read_text())
    source = json.loads(source_path.read_text())
    source_payload = {key: value for key, value in source.items() if key != "lock_payload_sha256"}
    inventory = json.loads((PROJECT_ROOT / source["inventory"]).read_text())
    historical = [json.loads((PROJECT_ROOT / path).read_text()) for path in config["historicalOutcomeLocks"]]
    selected = structurally_select(config, inventory)
    strata = Counter((row["service"], row["active_intent"]) for row in selected)
    requested = Counter({
        (item["service"], item["activeIntent"]): item["count"]
        for item in config["population"]["strata"]
    })
    models = config["modelConditions"]
    expected_models = {
        "qwen35_4b_4bit": ("mlx-community/Qwen3.5-4B-4bit", "0e7ffd5c629ef7719d4cbc04069232580bfa9d9c", 4),
        "qwen35_27b_4bit": ("mlx-community/Qwen3.5-27B-4bit", "45797d2985a12c55e6473686e9ea91b95e959553", 4),
        "qwen38_27b_4bit": ("mlx-community/Qwen3.8-27B-4bit", "3e6447f082e89cc7f0bc6e5441afd38dfce760ff", 4),
        "qwen38_27b_8bit": ("mlx-community/Qwen3.8-27B-8bit", "815b83c0df8ffd1d1b5244cf75fd6ef14fca9ef9", 8),
    }
    qg = config["qualityGatesPerCondition"]
    ag = config["accessGatesPerCondition"]
    stage = config["stageAuthorization"]
    checks = {
        "positive_fresh_source_outcome_is_exact_and_authorizes_design": bool(
            payload_hash(source_payload) == source["lock_payload_sha256"]
            and source["outcome"]["passed"]
            and source["authorization"]["preregister_fresh_hash_selected_multi_model_shadow_population"]
            and not source["authorization"]["download_or_run_model_before_population_prompt_and_gates_lock"]
            and file_sha256(PROJECT_ROOT / source["inventory"]) == source["inventory_sha256"]
        ),
        "V88r1_negative_and_V89_pause_history_is_preserved": bool(
            len(historical) == 2
            and not historical[0]["outcome"]["passed"]
            and historical[1]["outcome"]["passed"]
            and historical[1]["outcome"]["decision"] == "pause_external_local_model_integration"
        ),
        "fresh_population_is_feasible_balanced_and_dialogue_unique": bool(
            len(selected) == config["population"]["recordCount"] == 48
            and strata == requested
            and len({row["dialogue_id"] for row in selected}) == 48
            and sum(row["active_intent"] == "NONE" for row in selected) == 24
            and sum(row["active_intent"] != "NONE" for row in selected) == 24
            and len({row["service"] for row in selected}) == 3
            and len({row["active_intent"] for row in selected if row["active_intent"] != "NONE"}) == 5
        ),
        "four_model_identity_capacity_generation_quantization_matrix_is_exact": bool(
            len(models) == 4
            and all(
                (item["repository"], item["revision"], item["quantizationBits"]) == expected_models[item["id"]]
                and item["weightBytes"] > 0
                for item in models
            )
        ),
        "zero_weight_model_or_API_access_before_design_lock": all(
            config["preDesignModelExposure"][key] == 0 for key in (
                "modelWeightDownloadCount", "newModelSnapshotCount", "modelLoadCount",
                "modelGenerationCount", "LLMAPICallCount",
            )
        ),
        "prompt_output_and_decoding_are_identical_fail_closed_and_non_authoritative": bool(
            "always include NONE" in config["systemPrompt"]
            and config["outputContract"]["exactKeys"] == ["intent_candidates", "state_slot_key_candidates"]
            and config["outputContract"]["permanentlyNonDeployable"]
            and not config["outputContract"]["executable"]
            and config["decoding"]["temperature"] == 0.0
            and config["decoding"]["samplesPerRecord"] == 1
            and not config["decoding"]["enableThinking"]
            and not config["decoding"]["retryOnMalformedOutput"]
            and config["inputContract"]["samePromptForEveryCondition"]
        ),
        "quality_gates_preserve_or_exceed_V88_noncompensatory_thresholds": bool(
            qg["minimumExactJSONParseRate"] == 1.0
            and qg["minimumOntologyConformanceRate"] == 1.0
            and qg["minimumMandatoryNoneInclusionRate"] == 1.0
            and qg["minimumGoldActiveIntentCoverageRate"] >= 0.9
            and qg["minimumIntentCandidateSetExactRate"] >= 0.7
            and qg["minimumNoneOnlyIntentExactRate"] >= 0.75
            and qg["minimumStateSlotKeyRecallRate"] >= 0.75
            and qg["minimumStateSlotKeyExactRate"] >= 0.5
            and qg["maximumMeanIntentCandidateCount"] <= 2.0
        ),
        "per_condition_access_is_one_load_one_generation_per_record_and_zero_authority": bool(
            ag["requiredRecordCount"] == 48
            and ag["maximumModelLoadCount"] == 1
            and ag["maximumModelGenerationCount"] == 48
            and all(ag[key] == 0 for key in (
                "maximumLLMAPICallCount", "maximumAdapterTrainingRunCount",
                "maximumManualUtteranceInspectionCount", "maximumRealServiceCallCount",
                "maximumExternalSideEffectCount",
            ))
            and not config["decisionRule"]["passAuthorizesDirectActionOrExecution"]
            and not config["decisionRule"]["passAuthorizesBeliefAuthority"]
        ),
        "combination_is_posthoc_bounded_and_cannot_authorize_live_cascade": bool(
            config["comparisonProtocol"]["combinationDiagnostic"]["minimumActiveCoverageGain"] >= 0.05
            and config["comparisonProtocol"]["combinationDiagnostic"]["minimumSmallOnlyCorrectActiveCount"] >= 2
            and not config["comparisonProtocol"]["combinationDiagnostic"]["authorizesLiveCascade"]
        ),
        "design_stage_has_no_corpus_weight_inference_training_or_execution_authority": bool(
            stage["auditAndFreezeDesign"]
            and not stage["selectExtractAndSealCorpus"]
            and not stage["inspectUtterancesManually"]
            and not stage["downloadPinnedModelWeights"]
            and not stage["implementAndAuditRunner"]
            and not stage["runLocalModels"]
            and not stage["runAPIModel"]
            and not stage["trainAdapter"]
            and not stage["performRealServiceCall"]
            and not stage["performExternalSideEffect"]
        ),
    }
    passed = all(checks.values())
    audit = {
        "schema_version": "90-capacity-generation-design-audit",
        "experiment": "v90_capacity_generation_design_audit",
        "passed": passed,
        "decision": "freeze_design_and_authorize_one_fresh_corpus_seal" if passed else "reject_V90_model_comparison_design",
        "checks": checks,
        "selected_structural_summary": {
            "record_count": len(selected),
            "dialogue_count": len({row["dialogue_id"] for row in selected}),
            "strata": {f"{k[0]}::{k[1]}": v for k, v in sorted(strata.items())},
            "record_id_sha256": payload_hash({"ids": [row["record_id"] for row in selected]}),
        },
        "access": {
            "utterance_extraction_count": 0,
            "manual_utterance_inspection_count": 0,
            "model_weight_download_count": 0,
            "model_load_count": 0,
            "model_generation_count": 0,
            "LLM_API_call_count": 0,
            "real_service_call_count": 0,
            "external_side_effect_count": 0,
        },
        "prior_auditor_attempt": None if prior_auditor_attempt is None else {
            "artifact": str(prior_audit_path.relative_to(PROJECT_ROOT)),
            "decision": prior_auditor_attempt["decision"],
            "failed_check": "V88r1_and_V89_negative_history_is_preserved",
            "repair": "interpret V89 as a successful model-free audit whose registered decision pauses model integration",
            "design_population_model_prompt_decoding_gate_or_decision_change": False,
            "utterance_model_or_API_access_before_repair": False
        },
    }
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    if not passed:
        print(json.dumps(audit, indent=2, sort_keys=True))
        raise SystemExit(1)

    lock = {
        "schema_version": "90-capacity-generation-design-lock",
        "experiment": "v90_capacity_generation_design_lock",
        "design": str(design_path.relative_to(PROJECT_ROOT)),
        "design_sha256": file_sha256(design_path),
        "config_payload": config,
        "source_outcome_lock": str(source_path.relative_to(PROJECT_ROOT)),
        "source_outcome_lock_sha256": file_sha256(source_path),
        "source_inventory": source["inventory"],
        "source_inventory_sha256": source["inventory_sha256"],
        "historical_outcome_locks": {
            path: file_sha256(PROJECT_ROOT / path) for path in config["historicalOutcomeLocks"]
        },
        "plan": str(plan_path.relative_to(PROJECT_ROOT)),
        "plan_sha256": file_sha256(plan_path),
        "auditor": str(auditor_path.relative_to(PROJECT_ROOT)),
        "auditor_sha256": file_sha256(auditor_path),
        "builder": str(builder_path.relative_to(PROJECT_ROOT)),
        "builder_sha256": file_sha256(builder_path),
        "design_audit": str(audit_path.relative_to(PROJECT_ROOT)),
        "design_audit_sha256": file_sha256(audit_path),
        "authorization": {
            "modify_population_models_prompt_decoding_controls_gates_or_decisions": False,
            "select_extract_and_seal_corpus_once": True,
            "manually_inspect_source_language": False,
            "download_model_weights_before_corpus_seal": False,
            "implement_and_audit_acquisition_and_runner_after_corpus_seal": True,
            "run_local_model_before_implementation_lock": False,
            "run_API_model_or_train_adapter": False,
            "grant_model_belief_action_or_execution_authority": False,
            "perform_real_service_call_or_external_side_effect": False,
        },
    }
    lock["lock_payload_sha256"] = payload_hash(lock)
    lock_path.write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n")
    print(json.dumps(audit, indent=2, sort_keys=True))
    print(json.dumps({"lock": str(lock_path.relative_to(PROJECT_ROOT)), "sha256": file_sha256(lock_path)}, indent=2))


if __name__ == "__main__":
    main()
