#!/usr/bin/env python3
"""Verify and freeze the V89 model-free failure decomposition."""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from typing import Any

from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT


def payload_hash(value: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def close(a: float, b: float, tolerance: float = 1e-12) -> bool:
    return abs(float(a) - float(b)) <= tolerance


def main() -> None:
    impl_path = PROJECT_ROOT / "configs/v89-model-free-failure-decomposition-implementation-lock.json"
    result_path = PROJECT_ROOT / "outputs/v89-model-free-failure-decomposition/evaluation/result.json"
    verifier_path = PROJECT_ROOT / "python/verify_and_freeze_v89_failure_decomposition_outcome.py"
    doc_path = PROJECT_ROOT / "docs/v89-model-free-failure-decomposition-results.md"
    audit_path = PROJECT_ROOT / "outputs/v89-model-free-failure-decomposition/outcome-audit.json"
    lock_path = PROJECT_ROOT / "configs/v89-model-free-failure-decomposition-outcome-lock.json"
    if audit_path.exists() or lock_path.exists(): raise RuntimeError("V89 outcome already frozen")
    impl = json.loads(impl_path.read_text()); impl_payload = {k: v for k, v in impl.items() if k != "lock_payload_sha256"}
    parent = json.loads((PROJECT_ROOT / impl["parent_outcome_lock"]).read_text())
    parent_result = json.loads((PROJECT_ROOT / impl["parent_result"]).read_text())
    result = json.loads(result_path.read_text())
    rows = [json.loads((PROJECT_ROOT / item["path"]).read_text()) for item in parent["raw_fixture_artifacts"]]
    malformed = [row for row in rows if not row["ontology_conformant"]]
    conforming = [row for row in rows if row["ontology_conformant"]]
    joint = Counter(
        f"intent_{'exact' if row['intent_candidate_exact'] else 'wrong'}__state_{'exact' if row['state_slot_key_exact'] else 'wrong'}"
        for row in conforming
    )
    malformed_roles = Counter(
        f"{row['service']}::{'NONE' if row['active_intent'] == 'NONE' else 'active'}" for row in malformed
    )
    strict_intent_exact = sum(row["intent_candidate_exact"] for row in rows)
    strict_state_exact = sum(row["state_slot_key_exact"] for row in rows)
    strict_active_covered = sum(row["gold_active_intent_covered"] for row in rows if row["active_intent"] != "NONE")
    malformed_active = sum(row["active_intent"] != "NONE" for row in malformed)
    serialization_upper = result["view_metrics"]["perfect_serialization_upper_bound"]
    serialization_state = result["view_metrics"]["perfect_serialization_plus_state_oracle"]
    diagnostic = result["diagnostics"]
    checks = {
        "implementation_lock_and_all_inputs_exact": bool(
            payload_hash(impl_payload) == impl["lock_payload_sha256"]
            and all(file_sha256(PROJECT_ROOT / impl[key]) == impl[f"{key}_sha256"] for key in (
                "design_lock", "parent_outcome_lock", "parent_result", "protocol", "evaluator", "implementation_auditor"
            ))
            and all(file_sha256(PROJECT_ROOT / item["path"]) == item["sha256"] for item in parent["raw_fixture_artifacts"])
        ),
        "strict_parent_metrics_reconstruct_exactly": bool(
            result["strict_parent_metrics_reconstructed"]
            and result["view_metrics"]["strict_frozen_outcome"] == parent_result["metrics"]
        ),
        "joint_and_malformed_diagnostics_reconstruct": bool(
            dict(sorted(joint.items())) == diagnostic["joint_exactness_among_conforming"]
            and dict(sorted(malformed_roles.items())) == diagnostic["malformed_by_service_and_label_role"]
            and len(conforming) == 36 and len(malformed) == 12
        ),
        "perfect_serialization_upper_bound_key_rates_reconstruct": bool(
            close(serialization_upper["intent_candidate_set_exact_rate"], (strict_intent_exact + len(malformed)) / 48)
            and close(serialization_upper["state_slot_key_exact_rate"], (strict_state_exact + len(malformed)) / 48)
            and close(serialization_upper["gold_active_intent_coverage_rate"], (strict_active_covered + malformed_active) / 24)
            and close(serialization_upper["intent_candidate_set_exact_rate"], 0.8125)
            and close(serialization_upper["state_slot_key_exact_rate"], 0.4583333333333333)
            and close(serialization_upper["gold_active_intent_coverage_rate"], 0.7916666666666666)
        ),
        "serialization_only_and_serialization_plus_state_both_fail_registered_intent_coverage": bool(
            not result["view_gates"]["perfect_serialization_upper_bound"]["gold_active_intent_coverage"]
            and not result["view_gates"]["perfect_serialization_upper_bound"]["state_slot_key_exact"]
            and not result["view_gates"]["perfect_serialization_plus_state_oracle"]["gold_active_intent_coverage"]
            and close(serialization_state["state_slot_key_exact_rate"], 1.0)
            and not diagnostic["serialization_only_upper_bound_satisfies_all_semantic_gates"]
            and not diagnostic["serialization_plus_state_oracle_satisfies_all_intent_gates"]
        ),
        "decision_pauses_external_local_model_integration": result["passed"] and result["decision"] == "pause_external_local_model_integration",
        "zero_language_model_API_training_service_or_side_effect_access": all(value == 0 for value in result["access"].values()),
    }
    passed = all(checks.values())
    audit = {"schema_version": "89-model-free-failure-decomposition-outcome-audit", "experiment": "v89_failure_decomposition_outcome_audit", "passed": passed, "decision": "freeze_verified_V89_pause_decision" if passed else "reject_V89_outcome", "checks": checks, "verified_upper_bound": {"intent_candidate_set_exact_rate": serialization_upper["intent_candidate_set_exact_rate"], "gold_active_intent_coverage_rate": serialization_upper["gold_active_intent_coverage_rate"], "state_slot_key_exact_rate": serialization_upper["state_slot_key_exact_rate"], "state_slot_key_recall": serialization_upper["state_slot_key_recall"]}, "claim_boundary": result["claim_boundary"]}
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    if not passed: print(json.dumps(audit, indent=2, sort_keys=True)); raise SystemExit(1)
    lock = {"schema_version": "89-model-free-failure-decomposition-outcome-lock", "experiment": "v89_failure_decomposition_outcome_lock", "implementation_lock": str(impl_path.relative_to(PROJECT_ROOT)), "implementation_lock_sha256": file_sha256(impl_path), "result": str(result_path.relative_to(PROJECT_ROOT)), "result_sha256": file_sha256(result_path), "verifier": str(verifier_path.relative_to(PROJECT_ROOT)), "verifier_sha256": file_sha256(verifier_path), "audit": str(audit_path.relative_to(PROJECT_ROOT)), "audit_sha256": file_sha256(audit_path), "results_document": str(doc_path.relative_to(PROJECT_ROOT)), "results_document_sha256": file_sha256(doc_path), "outcome": {"passed": True, "decision": result["decision"], "diagnostics": result["diagnostics"], "serialization_upper_bound_metrics": serialization_upper}, "authorization": {"modify_or_rerun_V89": False, "pause_external_local_model_integration": True, "retain_verified_Bayesian_core_and_V83_V86_deterministic_interface": True, "access_local_or_API_model_for_this_branch": False, "train_adapter_or_learned_likelihood": False, "deploy_or_execute_any_language_model_output": False, "perform_real_service_call_or_external_side_effect": False, "report_and_synthesize_frozen_results": True}}
    lock["lock_payload_sha256"] = payload_hash(lock); lock_path.write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n")
    print(json.dumps(audit, indent=2, sort_keys=True)); print(json.dumps({"lock": str(lock_path), "sha256": file_sha256(lock_path)}, indent=2))


if __name__ == "__main__": main()
