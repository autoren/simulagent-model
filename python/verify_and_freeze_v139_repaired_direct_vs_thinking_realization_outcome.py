#!/usr/bin/env python3
from __future__ import annotations

import json
from typing import Any

from audit_and_freeze_v122_prequery_signal_inventory import payload_hash
from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT
from v139_repaired_direct_vs_thinking_realization import evaluate_experiment


def main() -> None:
    lock_path = PROJECT_ROOT / "configs/v139-repaired-direct-vs-thinking-realization-lock.json"
    result_path = PROJECT_ROOT / "outputs/v139-repaired-direct-vs-thinking-realization/model-realization/result.json"
    access_path = PROJECT_ROOT / "outputs/v139-repaired-direct-vs-thinking-realization/model-realization/access.json"
    doc_path = PROJECT_ROOT / "docs/v139-repaired-direct-vs-thinking-realization-results.md"
    audit_path = PROJECT_ROOT / "outputs/v139-repaired-direct-vs-thinking-realization/outcome-audit.json"
    outcome_path = PROJECT_ROOT / "configs/v139-repaired-direct-vs-thinking-realization-outcome-lock.json"
    verifier_path = PROJECT_ROOT / "python/verify_and_freeze_v139_repaired_direct_vs_thinking_realization_outcome.py"
    if audit_path.exists() or outcome_path.exists():
        raise RuntimeError("V139 outcome already frozen")
    if not doc_path.is_file():
        raise RuntimeError("write V139 results first")
    lock = json.loads(lock_path.read_text())
    config = lock["config_payload"]
    result = json.loads(result_path.read_text())
    access = json.loads(access_path.read_text())
    catalog = json.loads((PROJECT_ROOT / lock["choice_catalog"]).read_text())
    hidden = [row for row in json.loads((PROJECT_ROOT / lock["hidden_fixtures"]).read_text()) if row["split"] == config["population"]["split"]]
    v136 = json.loads((PROJECT_ROOT / lock["V136_config"]).read_text())
    expected_summary = evaluate_experiment(result["fixtures"], hidden, catalog, v136, access, config)
    dependencies = [key for key in lock if not key.endswith("_sha256") and f"{key}_sha256" in lock]
    raw_fields = {"raw_response", "thinking_trace", "final_text"}
    checks = {
        "lock_and_dependencies_exact": bool(
            payload_hash({key: value for key, value in lock.items() if key != "lock_payload_sha256"}) == lock["lock_payload_sha256"]
            and all(file_sha256(PROJECT_ROOT / lock[key]) == lock[f"{key}_sha256"] for key in dependencies)
        ),
        "completed_exact_comparison": bool(result["completed_condition"] and len(result["fixtures"]) == 200),
        "summary_exact": result["summary"] == expected_summary,
        "decision_exact": result["decision"] == expected_summary["decision"],
        "no_raw_response_or_trace_persisted": all(not (raw_fields & set(row)) and not row["raw_response_or_trace_persisted"] for row in result["fixtures"].values()),
        "all_outputs_non_authoritative": all(row["permanently_non_authoritative"] and not row["safe_hypothesis_universe_pruned"] and not row["capability_defined"] and not row["executable"] for row in result["fixtures"].values()),
        "access_and_zero_execution_pass": bool(expected_summary["access_pass"] and expected_summary["actual_execution_count"] == 0),
    }
    passed = all(checks.values())
    audit = {
        "schema_version": "139-repaired-direct-vs-thinking-realization-outcome-audit",
        "experiment": config["experiment"],
        "passed": passed,
        "checks": checks,
        "decision": expected_summary["decision"],
    }
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    if not passed:
        raise SystemExit(1)
    paths = {
        "analysis_lock": lock_path,
        "result": result_path,
        "access": access_path,
        "verifier": verifier_path,
        "audit": audit_path,
        "results_document": doc_path,
    }
    condition_summaries = {
        condition_id: {"qualified": row["qualified"], "metrics": row["metrics"], "gates": row["gates"]}
        for condition_id, row in expected_summary["conditions"].items()
    }
    outcome: dict[str, Any] = {
        "schema_version": "139-repaired-direct-vs-thinking-realization-outcome-lock",
        "experiment": "v139_repaired_direct_vs_thinking_realization_outcome_lock",
        "outcome": {
            "passed": True,
            "audit_pass": True,
            "comparison_completed": True,
            "at_least_one_condition_qualified": expected_summary["at_least_one_condition_qualified"],
            "decision": expected_summary["decision"],
            "conditions": condition_summaries,
        },
        "authorization": {
            "modify_retry_rerun_reprompt_or_mine_V139": False,
            "retain_as_synthetic_development_evidence_only": True,
            "preregister_separate_externally_authored_transfer_only_if_a_condition_qualified": expected_summary["at_least_one_condition_qualified"],
            "touch_V134_language_or_run_API": False,
            "run_induction_training_authority_action_or_execution": False,
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
