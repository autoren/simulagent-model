from __future__ import annotations

import json

from v92_architecture_audit import PROJECT_ROOT, build_audit, file_sha256, load_json, payload_sha256


DESIGN = "configs/v92-structured-llm-architecture-design.json"
AUDIT = "outputs/v92-structured-llm-architecture/architecture-audit.json"
LOCK = "configs/v92-structured-llm-architecture-outcome-lock.json"
RESULTS_DOC = "docs/v92-structured-llm-architecture-assessment.md"
RESEARCH_DIRECTION = "docs/research-direction.md"


def main() -> None:
    design = load_json(DESIGN)
    audit = build_audit(design)
    for required in (RESULTS_DOC, RESEARCH_DIRECTION):
        if not (PROJECT_ROOT / required).is_file():
            raise AssertionError(f"missing required synthesis document: {required}")

    audit_path = PROJECT_ROOT / AUDIT
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")

    lock = {
        "schema_version": "92-structured-llm-architecture-outcome-lock",
        "experiment": "v92_frozen_structured_llm_architecture_synthesis",
        "outcome": {
            "passed": True,
            "decision": audit["decision"],
            "runtime_architecture": audit["runtime_architecture"],
            "learned_role_qualification": audit["learned_role_qualification"],
            "cumulative_model_access": audit["cumulative_model_access"],
            "stopped_conditional_stages": audit["stopped_conditional_stages"],
            "stopping_reason": audit["stopping_reason"],
        },
        "design": DESIGN,
        "design_sha256": file_sha256(DESIGN),
        "audit": AUDIT,
        "audit_sha256": file_sha256(AUDIT),
        "verifier": "python/verify_and_freeze_v92_architecture.py",
        "verifier_sha256": file_sha256("python/verify_and_freeze_v92_architecture.py"),
        "protocol": "python/v92_architecture_audit.py",
        "protocol_sha256": file_sha256("python/v92_architecture_audit.py"),
        "results_document": RESULTS_DOC,
        "results_document_sha256": file_sha256(RESULTS_DOC),
        "research_direction": RESEARCH_DIRECTION,
        "research_direction_sha256": file_sha256(RESEARCH_DIRECTION),
        "authorization": {
            "retain_complete_deterministic_schema_enumeration_and_NONE": True,
            "retain_deterministic_state_validation_and_V86_surfaces": True,
            "retain_exact_posterior_aware_planning_and_execution_certificates": True,
            "use_any_local_or_API_model_in_runtime_decision_path": False,
            "use_Qwen35_4B_as_historical_shadow_baseline_only": True,
            "adopt_or_combine_any_27B_or_8bit_model": False,
            "run_API_capacity_comparator_for_current_branch": False,
            "train_adapter_or_learn_likelihood_for_current_branch": False,
            "prune_hypotheses_or_early_stop_from_model_ranking": False,
            "grant_model_state_belief_action_or_execution_authority": False,
            "perform_real_tool_or_service_call_or_external_side_effect": False,
            "report_frozen_bounded_results_with_claim_limits": True,
            "begin_future_LLM_study_only_with_materially_new_preregistered_role_and_fresh_evidence": True,
        },
    }
    lock["lock_payload_sha256"] = payload_sha256(lock)
    lock_path = PROJECT_ROOT / LOCK
    lock_path.write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n")

    print(json.dumps(audit, indent=2, sort_keys=True))
    print(json.dumps({"lock": LOCK, "sha256": file_sha256(LOCK)}, indent=2))


if __name__ == "__main__":
    main()
