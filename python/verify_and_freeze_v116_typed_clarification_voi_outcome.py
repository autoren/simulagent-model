#!/usr/bin/env python3
from __future__ import annotations

import json
from typing import Any

from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT
from v116_typed_clarification_voi import run_audit
from audit_and_freeze_v116_typed_clarification_voi import payload_hash
from run_v116_typed_clarification_voi import access_gates


def main() -> None:
    lock_path = PROJECT_ROOT / "configs/v116-typed-clarification-voi-lock.json"
    result_path = PROJECT_ROOT / "outputs/v116-typed-clarification-voi/audit/result.json"
    doc_path = PROJECT_ROOT / "docs/v116-typed-clarification-voi-results.md"
    audit_path = PROJECT_ROOT / "outputs/v116-typed-clarification-voi/outcome-audit.json"
    outcome_path = PROJECT_ROOT / "configs/v116-typed-clarification-voi-outcome-lock.json"
    verifier_path = PROJECT_ROOT / "python/verify_and_freeze_v116_typed_clarification_voi_outcome.py"
    if audit_path.exists() or outcome_path.exists():
        raise RuntimeError("V116 outcome is already frozen")
    if not doc_path.is_file():
        raise RuntimeError("write the V116 result document before freezing")
    lock = json.loads(lock_path.read_text())
    result = json.loads(result_path.read_text())
    population = json.loads((PROJECT_ROOT / lock["historical_population"]).read_text())
    historical = json.loads((PROJECT_ROOT / lock["historical_model_result"]).read_text())
    catalog = json.loads((PROJECT_ROOT / lock["choice_catalog"]).read_text())
    summary = run_audit(population, historical, catalog, lock["baseline_config_payload"], lock["config_payload"])
    gates = access_gates(result["access"], lock["config_payload"])
    dependency_keys = (
        "config", "parent_outcome", "parent_analysis_lock", "historical_population",
        "historical_model_result", "choice_catalog", "baseline_lock", "plan", "protocol",
        "tests", "runner", "verifier", "auditor", "design_audit",
    )
    checks = {
        "lock_and_dependencies_are_exact": bool(
            payload_hash({key: value for key, value in lock.items() if key != "lock_payload_sha256"}) == lock["lock_payload_sha256"]
            and all(file_sha256(PROJECT_ROOT / lock[key]) == lock[f"{key}_sha256"] for key in dependency_keys)
        ),
        "summary_and_decision_reconstruct_exactly": summary == result["summary"] and result["decision"] == summary["decision"],
        "all_access_gates_pass_with_zero_language_and_model_activity": bool(all(gates.values()) and all(value == 0 for value in result["access"].values())),
        "aggregate_only_complete_hypothesis_retention_and_zero_execution": bool(
            summary["individual_record_emission_count"] == 0
            and summary["true_hypothesis_retention"] == 1.0
            and summary["actual_execution_count"] == 0
        ),
        "no_expansive_authority_is_granted": bool(
            not lock["authorization"]["load_or_generate_with_model"]
            and not lock["authorization"]["begin_induction_or_richer_planning"]
            and not lock["authorization"]["grant_capability_belief_action_or_execution_authority"]
        ),
    }
    passed = all(checks.values())
    audit = {
        "schema_version": "116-typed-clarification-voi-outcome-audit",
        "experiment": lock["config_payload"]["experiment"], "passed": passed,
        "checks": checks, "independent_summary": summary,
    }
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    if not passed:
        print(json.dumps(audit, indent=2, sort_keys=True))
        raise SystemExit(1)
    deps = {"analysis_lock": lock_path, "result": result_path, "verifier": verifier_path, "audit": audit_path, "results_document": doc_path}
    outcome: dict[str, Any] = {
        "schema_version": "116-typed-clarification-voi-outcome-lock",
        "experiment": "v116_typed_clarification_voi_outcome_lock",
        "outcome": {"passed": True, "summary": summary, "decision": summary["decision"]},
        "authorization": {
            "modify_rerun_or_retune_V116": False,
            "preregister_unprotected_simulator_benchmark": bool(summary["independent_pass"]),
            "claim_human_or_model_answer_reliability": False,
            "read_fresh_or_protected_language": False, "run_model_or_API_or_training": False,
            "begin_schema_induction_or_richer_planning": False,
            "grant_capability_belief_action_or_execution_authority": False,
            "perform_real_service_call_or_external_side_effect": False,
        },
    }
    for key, path in deps.items():
        outcome[key] = str(path.relative_to(PROJECT_ROOT)); outcome[f"{key}_sha256"] = file_sha256(path)
    outcome["lock_payload_sha256"] = payload_hash(outcome)
    outcome_path.write_text(json.dumps(outcome, indent=2, sort_keys=True) + "\n")
    print(json.dumps(audit, indent=2, sort_keys=True))
    print(json.dumps({"lock": str(outcome_path.relative_to(PROJECT_ROOT)), "sha256": file_sha256(outcome_path)}, indent=2))


if __name__ == "__main__":
    main()
