#!/usr/bin/env python3
import json
from typing import Any

from audit_and_freeze_v122_prequery_signal_inventory import payload_hash
from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT
from v123_fresh_population_availability_audit import run_audit


def main() -> None:
    lock_path = PROJECT_ROOT / "configs/v123-fresh-population-availability-audit-lock.json"
    result_path = PROJECT_ROOT / "outputs/v123-fresh-population-availability-audit/audit/result.json"
    doc_path = PROJECT_ROOT / "docs/v123-fresh-population-availability-audit-results.md"
    audit_path = PROJECT_ROOT / "outputs/v123-fresh-population-availability-audit/outcome-audit.json"
    outcome_path = PROJECT_ROOT / "configs/v123-fresh-population-availability-audit-outcome-lock.json"
    verifier_path = PROJECT_ROOT / "python/verify_and_freeze_v123_fresh_population_availability_outcome.py"
    if audit_path.exists() or outcome_path.exists():
        raise RuntimeError("V123 outcome already frozen")
    if not doc_path.is_file():
        raise RuntimeError("write V123 result document first")
    lock = json.loads(lock_path.read_text())
    result = json.loads(result_path.read_text())
    inventory = json.loads((PROJECT_ROOT / lock["source_inventory"]).read_text())
    excluded = [json.loads((PROJECT_ROOT / lock[key]).read_text()) for key in sorted(key for key in lock if key.startswith("excluded_population_") and not key.endswith("_sha256"))]
    summary = run_audit(inventory, excluded, lock["config_payload"])
    dependencies = [key for key in lock if not key.endswith("_sha256") and f"{key}_sha256" in lock]
    checks = {
        "lock_and_dependencies_exact": bool(payload_hash({key: value for key, value in lock.items() if key != "lock_payload_sha256"}) == lock["lock_payload_sha256"] and all(file_sha256(PROJECT_ROOT / lock[key]) == lock[f"{key}_sha256"] for key in dependencies)),
        "summary_exact": summary == result["summary"] and result["passed"] == summary["outcome_pass"],
        "zero_language_model_execution_and_side_effects": all(value == 0 for value in result["access"].values()) and summary["language_read_count"] == 0 and summary["actual_execution_count"] == 0,
        "candidate_requirement_rejected_without_shrink": not summary["candidate_requirement_pass"] and summary["maximum_balanced_record_count_per_class"] == 9,
    }
    passed = all(checks.values())
    audit = {"schema_version": "123-fresh-population-availability-outcome-audit", "experiment": lock["config_payload"]["experiment"], "passed": passed, "checks": checks, "independent_summary": summary}
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    if not passed:
        raise SystemExit(1)
    paths = {"analysis_lock": lock_path, "result": result_path, "verifier": verifier_path, "audit": audit_path, "results_document": doc_path}
    outcome: dict[str, Any] = {
        "schema_version": "123-fresh-population-availability-outcome-lock",
        "experiment": "v123_fresh_population_availability_outcome_lock",
        "outcome": {"passed": True, "audit_pass": summary["outcome_pass"], "decision": summary["decision"], "summary": summary},
        "authorization": {
            "modify_rerun_or_shrink_V123": False,
            "preregister_external_controlled_open_set_source_feasibility_audit": bool(summary["outcome_pass"]),
            "reuse_prior_or_protected_identifiers": False,
            "evaluate_language_signal_trigger_or_model": False,
            "begin_induction_or_richer_planning": False,
            "run_API_training_action_or_execution": False,
        },
    }
    for key, path in paths.items():
        outcome[key] = str(path.relative_to(PROJECT_ROOT))
        outcome[f"{key}_sha256"] = file_sha256(path)
    outcome["lock_payload_sha256"] = payload_hash(outcome)
    outcome_path.write_text(json.dumps(outcome, indent=2, sort_keys=True) + "\n")
    print(json.dumps(audit, indent=2, sort_keys=True))
    print(json.dumps({"lock": str(outcome_path.relative_to(PROJECT_ROOT)), "sha256": file_sha256(outcome_path)}, indent=2))


if __name__ == "__main__":
    main()
