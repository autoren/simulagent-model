#!/usr/bin/env python3
from __future__ import annotations

import json
from typing import Any

from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT
from audit_and_freeze_v167_exact_evidence_gathering_planner import payload_hash
from run_v186_typed_contract_question_codebook_feasibility import DEPENDENCY_KEYS, reconstruct


def write_json(path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def main() -> None:
    lock_path = PROJECT_ROOT / "configs/v186-typed-contract-question-codebook-feasibility-lock.json"
    output_root = PROJECT_ROOT / "outputs/v186-typed-contract-question-codebook-feasibility/codebook"
    result_path = output_root / "result.json"
    doc_path = PROJECT_ROOT / "docs/v186-typed-contract-question-codebook-feasibility-results.md"
    verifier_path = PROJECT_ROOT / "python/verify_and_freeze_v186_typed_contract_question_codebook_feasibility_outcome.py"
    audit_path = PROJECT_ROOT / "outputs/v186-typed-contract-question-codebook-feasibility/outcome-audit.json"
    outcome_path = PROJECT_ROOT / "configs/v186-typed-contract-question-codebook-feasibility-outcome-lock.json"
    if audit_path.exists() or outcome_path.exists():
        raise RuntimeError("V186 outcome is already frozen")
    if not doc_path.is_file():
        raise RuntimeError("write V186 results before freezing")
    lock = json.loads(lock_path.read_text())
    result = json.loads(result_path.read_text())
    codebook, codebook_audit = reconstruct(lock)
    expected = {
        "question_codebook": {"questions": codebook["questions"], "source": "frozen semantic contract payload only"},
        "contract_answer_vectors": codebook["contract_answer_vectors"],
        "equivalence_classes": {"classes": codebook["equivalence_classes"]},
        "pairwise_separation": {"pairs": codebook["pairwise_separation"]},
        "development_bindings": {"bindings": codebook["bindings"]["development"], "contains_language": False},
        "protected_bindings": {"bindings": codebook["bindings"]["protected"], "contains_language": False},
        "codebook_summary": codebook["summary"],
    }
    outputs_exact = all(
        json.loads((PROJECT_ROOT / result["output_integrity"][key]["path"]).read_text()) == payload
        and file_sha256(PROJECT_ROOT / result["output_integrity"][key]["path"])
        == result["output_integrity"][key]["sha256"]
        for key, payload in expected.items()
    )
    config = lock["config_payload"]
    expected_decision = (
        config["decisionRule"]["ifEveryCodebookSeparabilityBindingAndSafetyGatePasses"]
        if codebook_audit["passed"]
        else config["decisionRule"]["otherwise"]
    )
    zero_keys = (
        "utterance_or_dialogue_language_read_count", "planner_policy_score_count", "model_load_count",
        "model_generation_count", "API_call_count", "training_run_count", "ontology_registration_count",
        "trusted_state_mutation_count", "service_call_count", "external_side_effect_count", "actual_execution_count",
    )
    checks = {
        "lock_and_dependencies_are_exact": bool(
            payload_hash({key: value for key, value in lock.items() if key != "lock_payload_sha256"})
            == lock["lock_payload_sha256"]
            and all(file_sha256(PROJECT_ROOT / lock[key]) == lock[f"{key}_sha256"] for key in DEPENDENCY_KEYS)
        ),
        "all_codebook_outputs_reconstruct_exactly": outputs_exact,
        "summary_gates_and_decision_reconstruct_exactly": bool(
            result["summary"] == codebook["summary"]
            and result["feasibility_gates"] == codebook_audit["checks"]
            and result["passed"] == codebook_audit["passed"]
            and result["decision"] == expected_decision
        ),
        "equivalence_and_role_binding_boundaries_hold": bool(
            codebook["summary"]["pairwise_separation_rate"] == 1.0
            and codebook["summary"]["largest_equivalence_class_size"] == 1
            and codebook["summary"]["role_identifier_overlap"] == 0
        ),
        "language_planner_model_authority_and_effect_access_is_zero": all(result["access"][key] == 0 for key in zero_keys),
    }
    verified = all(checks.values())
    outcome_audit = {
        "schema_version": "186-typed-contract-question-codebook-feasibility-outcome-audit",
        "experiment": config["experiment"],
        "passed": verified,
        "scientific_feasibility_gates_passed": codebook_audit["passed"],
        "decision": "freeze_verified_V186_codebook" if verified else "reject_V186_outcome",
        "checks": checks,
        "independent_summary": codebook["summary"],
        "additional_access": {
            "utterance_or_dialogue_language_read_count": 0,
            "planner_policy_score_count": 0,
            "model_load_count": 0,
            "actual_execution_count": 0,
        },
    }
    write_json(audit_path, outcome_audit)
    if not verified:
        print(json.dumps(outcome_audit, indent=2, sort_keys=True))
        raise SystemExit(1)
    dependencies = {
        "codebook_lock": lock_path,
        "result": result_path,
        "verifier": verifier_path,
        "audit": audit_path,
        "results_document": doc_path,
        "parent_V185_outcome": PROJECT_ROOT / lock["parent_V185_outcome"],
    }
    for key, item in result["output_integrity"].items():
        dependencies[key] = PROJECT_ROOT / item["path"]
    outcome: dict[str, Any] = {
        "schema_version": "186-typed-contract-question-codebook-feasibility-outcome-lock",
        "experiment": config["experiment"],
        "outcome": {
            "passed": True,
            "scientific_feasibility_gates_passed": codebook_audit["passed"],
            "decision": result["decision"],
            "summary": result["summary"],
        },
        "authorization": {
            "modify_rebuild_or_redefine_V186": False,
            "preregister_clean_exact_planner_comparison": bool(codebook_audit["passed"]),
            "score_planner_without_separate_lock": False,
            "read_utterance_or_protected_language_run_model_API_or_training": False,
            "register_mutate_call_service_act_or_execute": False,
        },
    }
    for key, path in dependencies.items():
        outcome[key] = str(path.relative_to(PROJECT_ROOT))
        outcome[f"{key}_sha256"] = file_sha256(path)
    outcome["lock_payload_sha256"] = payload_hash(outcome)
    write_json(outcome_path, outcome)
    print(json.dumps(outcome_audit, indent=2, sort_keys=True))
    print(json.dumps({"lock": str(outcome_path.relative_to(PROJECT_ROOT)), "sha256": file_sha256(outcome_path)}, indent=2))


if __name__ == "__main__":
    main()
