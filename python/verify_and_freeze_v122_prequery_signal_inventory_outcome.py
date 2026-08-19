#!/usr/bin/env python3
import json
from typing import Any

from audit_and_freeze_v122_prequery_signal_inventory import payload_hash
from v10_protocol import file_sha256
from v122_prequery_signal_inventory import build_inventory
from v22r2_grounding import PROJECT_ROOT


def main() -> None:
    lock_path = PROJECT_ROOT / "configs/v122-prequery-signal-inventory-lock.json"
    result_path = PROJECT_ROOT / "outputs/v122-prequery-signal-inventory/audit/result.json"
    doc_path = PROJECT_ROOT / "docs/v122-prequery-signal-inventory-results.md"
    audit_path = PROJECT_ROOT / "outputs/v122-prequery-signal-inventory/outcome-audit.json"
    outcome_path = PROJECT_ROOT / "configs/v122-prequery-signal-inventory-outcome-lock.json"
    verifier_path = PROJECT_ROOT / "python/verify_and_freeze_v122_prequery_signal_inventory_outcome.py"
    if audit_path.exists() or outcome_path.exists():
        raise RuntimeError("V122 outcome already frozen")
    if not doc_path.is_file():
        raise RuntimeError("write V122 result document first")

    lock = json.loads(lock_path.read_text())
    result = json.loads(result_path.read_text())
    summary = build_inventory(lock["config_payload"])
    dependencies = [key for key in lock if not key.endswith("_sha256") and f"{key}_sha256" in lock]
    checks = {
        "lock_and_dependencies_exact": bool(
            payload_hash({key: value for key, value in lock.items() if key != "lock_payload_sha256"}) == lock["lock_payload_sha256"]
            and all(file_sha256(PROJECT_ROOT / lock[key]) == lock[f"{key}_sha256"] for key in dependencies)
        ),
        "summary_exact": summary == result["summary"] and result["passed"] == summary["outcome_pass"],
        "zero_access_evaluation_fit_and_execution": bool(
            all(value == 0 for value in result["access"].values())
            and summary["signal_evaluated_count"] == 0
            and summary["trigger_fitted_count"] == 0
            and summary["actual_execution_count"] == 0
        ),
        "retrieval_geometry_is_only_llm_independent_semantic_family": summary["llm_independent_semantic_families"] == ["retrieval_geometry"],
    }
    passed = all(checks.values())
    audit = {
        "schema_version": "122-prequery-signal-inventory-outcome-audit",
        "experiment": lock["config_payload"]["experiment"],
        "passed": passed,
        "checks": checks,
        "independent_summary": summary,
    }
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    if not passed:
        raise SystemExit(1)

    paths = {
        "analysis_lock": lock_path,
        "result": result_path,
        "verifier": verifier_path,
        "audit": audit_path,
        "results_document": doc_path,
    }
    outcome: dict[str, Any] = {
        "schema_version": "122-prequery-signal-inventory-outcome-lock",
        "experiment": "v122_prequery_signal_inventory_outcome_lock",
        "outcome": {
            "passed": True,
            "audit_pass": summary["outcome_pass"],
            "decision": summary["decision"],
            "summary": summary,
        },
        "authorization": {
            "modify_rerun_or_retune_V122": False,
            "preregister_fresh_model_free_retrieval_geometry_design": bool(summary["outcome_pass"]),
            "evaluate_signals_or_fit_trigger": False,
            "run_language_model_or_protected": False,
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
