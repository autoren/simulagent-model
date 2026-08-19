from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_json(relative_path: str) -> dict[str, Any]:
    return json.loads((PROJECT_ROOT / relative_path).read_text())


def file_sha256(relative_path: str) -> str:
    return hashlib.sha256((PROJECT_ROOT / relative_path).read_bytes()).hexdigest()


def payload_sha256(payload: dict[str, Any]) -> str:
    normalized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(normalized.encode()).hexdigest()


def verify_internal_lock(relative_path: str) -> dict[str, Any]:
    payload = load_json(relative_path)
    expected = payload.get("lock_payload_sha256")
    if expected is None:
        raise AssertionError(f"missing lock_payload_sha256: {relative_path}")
    unhashed = copy.deepcopy(payload)
    unhashed.pop("lock_payload_sha256")
    if payload_sha256(unhashed) != expected:
        raise AssertionError(f"payload hash mismatch: {relative_path}")
    return payload


def access_count(payload: dict[str, Any], *keys: str) -> int:
    for key in keys:
        value = payload.get(key)
        if value is not None:
            return int(value)
    return 0


def summarize_model_access(locks: dict[str, dict[str, Any]]) -> dict[str, int]:
    v80 = load_json(locks["v80"]["access"])
    v81 = load_json(locks["v81"]["access"])
    v82 = load_json(locks["v82"]["access"])
    v85 = load_json(locks["v85"]["access"])
    v88 = locks["v88r1"]["outcome"]["cumulative_access"]
    v90 = locks["v90"]["outcome"]["access"]
    v91 = locks["v91"]["outcome"]["access"]
    rows = [v80, v81, v82, v85, v88, v90, v91]
    return {
        "model_load_count": sum(access_count(row, "model_load_count") for row in rows),
        "model_generation_count": sum(
            access_count(row, "model_generation_count", "model_forward_pass_count")
            for row in rows
        ),
        "LLM_API_call_count": sum(
            access_count(row, "LLM_API_call_count", "API_call_count") for row in rows
        ),
        "adapter_training_run_count": sum(
            access_count(row, "adapter_training_run_count") for row in rows
        ),
        "external_side_effect_count": sum(
            access_count(row, "external_side_effect_count") for row in rows
        ),
        "real_tool_or_service_call_count": sum(
            access_count(row, "real_tool_call_count", "real_service_call_count")
            for row in rows
        ),
    }


def build_audit(design: dict[str, Any]) -> dict[str, Any]:
    evidence_paths = design["evidenceLocks"]
    evidence = {path: verify_internal_lock(path) for path in evidence_paths}
    locks = {
        "v77": evidence["configs/v77-execution-closure-lock.json"],
        "v78": evidence["configs/v78-clarification-outcome-lock.json"],
        "v79": evidence["configs/v79-terminal-utility-outcome-lock.json"],
        "v80": evidence["configs/v80-local-candidate-generation-outcome-lock.json"],
        "v81": evidence["configs/v81-factorized-local-candidate-outcome-lock.json"],
        "v82": evidence["configs/v82-local-clarification-surface-outcome-lock.json"],
        "v83": evidence["configs/v83-strict-clarification-interface-outcome-lock.json"],
        "v84": evidence["configs/v84-schema-grounded-shadow-outcome-lock.json"],
        "v85": evidence["configs/v85-local-adversarial-generator-outcome-lock.json"],
        "v86": evidence["configs/v86-partial-option-validator-outcome-lock.json"],
        "v87": evidence["configs/v87-external-language-source-outcome-lock.json"],
        "v88r1": evidence["configs/v88r1-external-intent-candidate-outcome-lock.json"],
        "v89": evidence["configs/v89-model-free-failure-decomposition-outcome-lock.json"],
        "v90": evidence["configs/v90-capacity-generation-outcome-lock.json"],
        "v91": evidence["configs/v91-rank-only-outcome-lock.json"],
    }

    result79 = load_json(locks["v79"]["result"])
    result78 = load_json(locks["v78"]["result"])
    metrics88 = locks["v88r1"]["outcome"]["metrics"]
    metrics91 = locks["v91"]["outcome"]["metrics"]
    v79_config = load_json(locks["v79"]["implementation_lock"])["resolved_config_payload"]

    learned_role_outcomes = {
        "synthetic_candidate_generation": locks["v80"]["outcome"]["passed"],
        "factorized_semantic_compatibility": locks["v81"]["outcome"]["passed"],
        "clarification_surface_rendering": locks["v82"]["outcome"]["passed"],
        "offline_adversarial_generation": locks["v85"]["outcome"]["passed"],
        "external_intent_and_state_generation": locks["v88r1"]["outcome"]["passed"],
        "capacity_or_precision_condition": bool(locks["v90"]["outcome"]["qualifying_27b_conditions"]),
        "rank_only_search_scheduling": locks["v91"]["outcome"]["condition_qualified"],
    }
    model_access = summarize_model_access(locks)
    v79_controls = set(v79_config["controls"])
    v88_controls = set(metrics88["controls"])
    v91_controls = set(metrics91["controls"])
    control_coverage = {
        "exhaustive_enumeration": "exhaustive" in v88_controls and "exhaustive_unordered" in v91_controls,
        "grammar": "identifier_exact_match_grammar" in v91_controls,
        "retrieval": "lexical_overlap" in v91_controls,
        "direct_action": "act_immediately" in v79_controls,
        "MAP": "map_interpretation" in v79_controls,
        "ask_always": "ask_always" in v79_controls,
        "act_immediately": "act_immediately" in v79_controls,
        "oracle_candidate": "oracle" in v88_controls,
        "oracle_interpretation": "oracle_interpretation" in v79_controls,
    }

    checks = {
        "all_evidence_locks_have_valid_internal_payload_hashes": len(evidence) == 15,
        "V77_is_execution_inconclusive_not_scientific_evidence": locks["v77"]["authorization"]["claim_v77_scientific_outcome"] is False,
        "V78_negative_design_result_is_preserved": result78["passed"] is False,
        "V79_model_free_benchmark_passed_every_gate": result79["passed"] is True and all(result79["gates"].values()),
        "every_registered_control_family_is_present": all(control_coverage.values()),
        "no_learned_role_qualified": not any(learned_role_outcomes.values()),
        "V83_V84_V86_model_free_boundaries_passed": all(locks[key]["outcome"]["passed"] for key in ("v83", "v84", "v86")),
        "external_language_source_audit_passed_without_text_inventory": locks["v87"]["outcome"]["passed"] is True and locks["v87"]["outcome"]["inventory_summary"]["contains_utterance_or_text_fields"] is False,
        "external_language_model_result_remains_non_deployable": metrics88["permanent_non_deployable_rate"] == 1.0 and locks["v88r1"]["authorization"]["deploy_or_execute_any_model_output"] is False,
        "capacity_comparison_resolved_without_API_or_adapter": len(locks["v90"]["outcome"]["condition_summaries"]) == 4 and not locks["v90"]["outcome"]["qualifying_27b_conditions"] and locks["v90"]["authorization"]["run_API_model_or_train_adapter_for_this_branch"] is False,
        "rank_only_wrapper_preserved_complete_set_NONE_state_and_planner": metrics91["canonical_complete_set_rate"] == 1.0 and metrics91["canonical_NONE_retention_rate"] == 1.0 and metrics91["authoritative_state_preservation_rate"] == 1.0 and locks["v91"]["outcome"]["planner_invariance"]["invariance_rate"] == 1.0,
        "rank_only_model_lost_to_best_deterministic_control": metrics91["MRR_improvement_over_best_nonoracle_control"] < 0 and metrics91["mean_rank_reduction_versus_best_nonoracle_control"] < 0,
        "all_model_access_was_local_frozen_and_non_authoritative": model_access == {"model_load_count": 11, "model_generation_count": 401, "LLM_API_call_count": 0, "adapter_training_run_count": 0, "external_side_effect_count": 0, "real_tool_or_service_call_count": 0},
        "current_authorization_retains_model_free_runtime": locks["v90"]["authorization"]["retain_model_free_authoritative_boundary"] is True and locks["v91"]["authorization"]["retain_complete_deterministic_schema_enumeration"] is True and locks["v91"]["authorization"]["use_local_model_as_candidate_generator_or_search_scheduler"] is False,
        "conditional_expansions_remain_unauthorized": locks["v91"]["authorization"]["add_larger_local_or_API_model_for_this_branch"] is False and locks["v91"]["authorization"]["train_adapter_or_learn_likelihood_for_this_branch"] is False and locks["v91"]["authorization"]["grant_model_belief_action_or_execution_authority"] is False,
    }
    if not all(checks.values()):
        failed = [name for name, passed in checks.items() if not passed]
        raise AssertionError(f"V92 synthesis checks failed: {failed}")

    return {
        "schema_version": "92-structured-llm-architecture-audit",
        "experiment": "v92_frozen_structured_llm_architecture_synthesis",
        "passed": True,
        "decision": "freeze_model_free_authoritative_runtime_and_close_current_structured_LLM_branch",
        "checks": checks,
        "control_coverage": control_coverage,
        "learned_role_qualification": learned_role_outcomes,
        "cumulative_model_access": model_access,
        "runtime_architecture": design["expectedArchitecture"],
        "stopped_conditional_stages": [
            "model_conditioned_posterior_or_likelihood_integration",
            "model_generated_mechanic_or_action_hypotheses",
            "downstream_model_conditioned_action_regret",
            "API_capacity_comparison",
            "adapter_or_LoRA_training",
            "learned_likelihood_scoring",
            "LLM_guided_pruning_or_early_stopping",
            "bounded_or_real_execution_authority",
        ],
        "stopping_reason": "all learned roles failed prerequisite noncompensatory utility gates; running later stages would violate the registered authorization chain",
        "claim_boundary": design["claimBoundary"],
        "evidence_hashes": {path: file_sha256(path) for path in evidence_paths},
    }
