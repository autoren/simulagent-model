#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
from typing import Any

from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT
from run_v112_open_world_full_policy_transfer import payload_hash
from run_v112r1_full_policy_aggregation import reconstruct, recovery_decision


def main() -> None:
    lock_path = PROJECT_ROOT / "configs/v112r1-full-policy-aggregation-recovery-lock.json"
    result_path = PROJECT_ROOT / "outputs/v112r1-full-policy-aggregation-recovery/recovered-evaluation/result.json"
    doc_path = PROJECT_ROOT / "docs/v112-open-world-full-policy-transfer-results.md"
    verifier_path = PROJECT_ROOT / "python/verify_and_freeze_v112r1_aggregation_outcome.py"
    audit_path = PROJECT_ROOT / "outputs/v112r1-full-policy-aggregation-recovery/outcome-audit.json"
    outcome_path = PROJECT_ROOT / "configs/v112r1-full-policy-aggregation-recovery-outcome-lock.json"
    if audit_path.exists() or outcome_path.exists():
        raise RuntimeError("V112r1 outcome is already frozen")
    if not doc_path.is_file():
        raise RuntimeError("write the V112/V112r1 result document before freezing")
    lock = json.loads(lock_path.read_text())
    result = json.loads(result_path.read_text())
    summary, reconstruction = reconstruct(lock)
    quality, novel, access, decision = recovery_decision(summary)
    dependency_keys = (
        "config", "parent_lock", "parent_failure", "fresh_language", "fixture_manifest",
        "source_archive", "visible_catalog", "fresh_population", "plan", "protocol", "tests",
        "runner", "verifier", "auditor", "design_audit",
    )
    checks = {
        "lock_and_dependencies_are_exact": bool(
            payload_hash({key: value for key, value in lock.items() if key != "lock_payload_sha256"}) == lock["lock_payload_sha256"]
            and all(file_sha256(PROJECT_ROOT / lock[key]) == lock[f"{key}_sha256"] for key in dependency_keys)
        ),
        "summary_and_decision_reconstruct_exactly": bool(
            summary == result["summary"] and reconstruction == result["reconstruction"]
            and quality == result["quality_gate_pass"] and novel == result["novel_evidence_pass"]
            and access == result["passed"] and decision == result["decision"]
        ),
        "all_240_preserved_fixtures_reused_without_model": bool(
            reconstruction["fixture_count"] == 240
            and result["access"]["preserved_fixture_automatic_read_count"] == 240
            and result["access"]["new_model_load_count"] == 0
            and result["access"]["new_model_generation_count"] == 0
        ),
        "zero_protected_manual_API_training_service_effect_and_execution": bool(
            all(result["access"][key] == 0 for key in (
                "protected_test_language_read_count", "manual_language_or_raw_response_inspection_count",
                "LLM_API_call_count", "adapter_training_run_count", "real_service_call_count",
                "external_side_effect_count",
            )) and summary["actual_execution_count"] == 0 and summary["true_hypothesis_retention"] == 1.0
        ),
    }
    passed = all(checks.values())
    audit = {
        "schema_version": "112r1-full-policy-aggregation-recovery-outcome-audit",
        "experiment": lock["config_payload"]["experiment"], "passed": passed,
        "quality_gate_pass": quality, "novel_evidence_pass": novel,
        "decision": decision, "checks": checks, "independent_summary": summary,
    }
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    if not passed:
        print(json.dumps(audit, indent=2, sort_keys=True))
        raise SystemExit(1)
    dependencies = {
        "analysis_lock": lock_path, "result": result_path, "verifier": verifier_path,
        "audit": audit_path, "results_document": doc_path,
    }
    outcome: dict[str, Any] = {
        "schema_version": "112r1-full-policy-aggregation-recovery-outcome-lock",
        "experiment": "v112r1_full_policy_aggregation_recovery_outcome_lock",
        "outcome": {
            "passed": True, "quality_gate_pass": quality,
            "novel_evidence_pass": novel, "decision": decision, "summary": summary,
        },
        "authorization": {
            "modify_rerun_retry_or_retune_V112_or_V112r1": False,
            "preregister_protected_test_confirmation": bool(quality),
            "seek_new_contrastive_or_multiturn_evidence": not novel,
            "redesign_policy_on_new_population": bool(novel and not quality),
            "read_protected_test_before_separate_lock": False,
            "proceed_to_schema_or_mechanic_induction": False,
            "proceed_to_richer_sequential_decision_problem": False,
            "run_additional_local_or_API_model_or_train_adapter": False,
            "prune_hypotheses_define_capability_or_grant_belief_action_execution_authority": False,
            "perform_real_service_call_or_external_side_effect": False,
        },
    }
    for key, path in dependencies.items():
        outcome[key] = str(path.relative_to(PROJECT_ROOT))
        outcome[f"{key}_sha256"] = file_sha256(path)
    outcome["lock_payload_sha256"] = hashlib.sha256(
        json.dumps(outcome, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    outcome_path.write_text(json.dumps(outcome, indent=2, sort_keys=True) + "\n")
    print(json.dumps(audit, indent=2, sort_keys=True))
    print(json.dumps({"lock": str(outcome_path.relative_to(PROJECT_ROOT)), "sha256": file_sha256(outcome_path)}, indent=2))


if __name__ == "__main__":
    main()
