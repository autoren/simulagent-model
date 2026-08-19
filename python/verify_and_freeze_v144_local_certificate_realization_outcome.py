#!/usr/bin/env python3
from __future__ import annotations

import json
from typing import Any

from audit_and_freeze_v122_prequery_signal_inventory import payload_hash, write_json
from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT
from v144_local_certificate_realization import evaluate


def main() -> None:
    lock_path = PROJECT_ROOT / "configs/v144-local-certificate-realization-lock.json"
    result_path = PROJECT_ROOT / "outputs/v144-local-certificate-realization/model-realization/result.json"
    access_path = PROJECT_ROOT / "outputs/v144-local-certificate-realization/model-realization/access.json"
    doc_path = PROJECT_ROOT / "docs/v144-local-certificate-realization-results.md"
    audit_path = PROJECT_ROOT / "outputs/v144-local-certificate-realization/outcome-audit.json"
    outcome_path = PROJECT_ROOT / "configs/v144-local-certificate-realization-outcome-lock.json"
    verifier_path = PROJECT_ROOT / "python/verify_and_freeze_v144_local_certificate_realization_outcome.py"
    if audit_path.exists() or outcome_path.exists():
        raise RuntimeError("V144 outcome already frozen")
    if not doc_path.is_file():
        raise RuntimeError("write the V144 results document before freezing the outcome")
    lock = json.loads(lock_path.read_text())
    config = lock["config_payload"]
    result = json.loads(result_path.read_text())
    access = json.loads(access_path.read_text())
    catalog = json.loads((PROJECT_ROOT / lock["choice_catalog"]).read_text())
    hidden = json.loads((PROJECT_ROOT / lock["development_hidden_fixtures"]).read_text())
    v136 = json.loads((PROJECT_ROOT / lock["V136_config"]).read_text())
    expected = evaluate(result["fixtures"], hidden, catalog, v136, access, config)
    dependencies = [key for key in lock if not key.endswith("_sha256") and f"{key}_sha256" in lock]
    raw_fields = {"raw_response", "thinking_trace", "final_text"}
    invalid = [row for row in result["fixtures"].values() if not row["certificate_valid"]]
    checks = {
        "preregistration_lock_and_dependencies_exact": bool(
            payload_hash({key: value for key, value in lock.items() if key != "lock_payload_sha256"}) == lock["lock_payload_sha256"]
            and all(file_sha256(PROJECT_ROOT / lock[key]) == lock[f"{key}_sha256"] for key in dependencies)
        ),
        "completed_exact_development_realization": bool(
            result["completed_condition"]
            and len(result["fixtures"]) == config["population"]["fixtureCount"]
            and access["model_generation_count"] == config["population"]["fixtureCount"]
        ),
        "summary_and_decision_exact": result["summary"] == expected and result["decision"] == expected["decision"],
        "zero_test_generation": access["test_fixture_model_generation_count"] == 0,
        "no_raw_response_trace_or_final_text_persisted": all(
            not (raw_fields & set(row)) and not row["raw_response_or_trace_persisted"]
            for row in result["fixtures"].values()
        ),
        "only_valid_normalized_certificates_persisted": all(
            (row["normalized_certificate"] is not None) == row["certificate_valid"]
            for row in result["fixtures"].values()
        ),
        "invalid_certificates_fail_closed": all(
            row["final_choice_id"] == "A00" and row["final_output_structurally_valid"]
            for row in invalid
        ),
        "all_outputs_non_authoritative_nonexecuting": all(
            row["permanently_non_authoritative"]
            and not row["authoritative_hypothesis_universe_pruned"]
            and not row["capability_defined"]
            and not row["executable"]
            and row["actual_execution_count"] == 0
            for row in result["fixtures"].values()
        ),
        "all_access_gates_pass": all(expected["access_gates"].values()),
    }
    passed = all(checks.values())
    audit = {
        "schema_version": "144-local-certificate-realization-outcome-audit",
        "experiment": config["experiment"],
        "passed": passed,
        "checks": checks,
        "qualified": expected["qualified"],
        "decision": expected["decision"],
        "metrics": expected["metrics"],
        "qualification_gates": expected["qualification_gates"],
        "access_gates": expected["access_gates"],
    }
    write_json(audit_path, audit)
    if not passed:
        print(json.dumps(audit, indent=2, sort_keys=True))
        raise SystemExit(1)
    paths = {
        "analysis_lock": lock_path,
        "result": result_path,
        "access": access_path,
        "verifier": verifier_path,
        "audit": audit_path,
        "results_document": doc_path,
    }
    outcome: dict[str, Any] = {
        "schema_version": "144-local-certificate-realization-outcome-lock",
        "experiment": config["experiment"],
        "outcome": {
            "passed": True,
            "audit_pass": True,
            "realization_completed": True,
            "qualified": expected["qualified"],
            "decision": expected["decision"],
            "metrics": expected["metrics"],
            "qualification_gates": expected["qualification_gates"],
        },
        "authorization": {
            "retain_as_project_authored_synthetic_development_evidence_only": True,
            "preregister_one_separate_frozen_V142_test_realization": expected["qualified"],
            "run_or_open_V142_test_before_separate_preregistration": False,
            "modify_retry_rerun_reprompt_tune_or_mine_V144": False,
            "touch_V134_external_language_or_run_API": False,
            "run_training_induction_authority_action_or_execution": False,
        },
    }
    for key, path in paths.items():
        outcome[key] = str(path.relative_to(PROJECT_ROOT))
        outcome[f"{key}_sha256"] = file_sha256(path)
    outcome["lock_payload_sha256"] = payload_hash(outcome)
    write_json(outcome_path, outcome)
    print(json.dumps(audit, indent=2, sort_keys=True))
    print(json.dumps({"lock": str(outcome_path.relative_to(PROJECT_ROOT)), "sha256": file_sha256(outcome_path)}, indent=2))


if __name__ == "__main__":
    main()
