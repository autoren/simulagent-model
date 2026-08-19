#!/usr/bin/env python3
from __future__ import annotations

import json
from typing import Any

from audit_and_freeze_v122_prequery_signal_inventory import payload_hash, write_json
from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT


def aggregate_checks(result, config):
    gates = config["outcomeGates"]
    conditions = result["summary"]["conditions"]
    selective = [row["selective_query"] for row in conditions.values()]
    return {
        "selective_regret_every_prior_and_correlation": all(row["mean_regret"] <= gates["maximumSelectiveMeanRegretEveryPriorAndCorrelation"] for row in selective),
        "selective_known_exact_every_prior_and_correlation": all(row["known_exact_probability"] >= gates["minimumSelectiveKnownExactEveryPriorAndCorrelation"] for row in selective),
        "selective_unsupported_every_prior_and_correlation": all(row["unsupported_correct_probability"] >= gates["minimumSelectiveUnsupportedCorrectEveryPriorAndCorrelation"] for row in selective),
        "selective_false_known_every_prior_and_correlation": all(row["false_known_probability"] <= gates["maximumSelectiveFalseKnownEveryPriorAndCorrelation"] for row in selective),
        "nontrivial_skip_fraction": all(gates["minimumSkipFraction"] <= row["skip_fraction"] <= gates["maximumSkipFraction"] for row in conditions.values()),
        "skipped_action_precision": all(row["skipped_action_precision"] >= gates["minimumSkippedActionPrecision"] for row in conditions.values()),
        "queried_average_value_covers_cost": all(row["queried_average_query_value"] >= gates["minimumQueriedAverageQueryValueEveryPriorAndCorrelation"] for row in conditions.values()),
        "skipped_average_value_not_above_cost": all(row["skipped_average_query_value"] <= gates["maximumSkippedAverageQueryValueEveryPriorAndCorrelation"] for row in conditions.values()),
        "selective_no_worse_than_always_query": all(row["selective_query"]["mean_regret"] <= row["always_query"]["mean_regret"] for row in conditions.values()),
        "one_rule_zero_fit_selection_and_thresholds": result["summary"]["primary_rule_count"] == 1 and result["summary"]["fit_count"] == result["summary"]["selection_count"] == result["summary"]["threshold_count"] == 0,
        "schema_signature_count": result["summary"]["schema_signature_count"] == gates["requiredSchemaSignatureCount"],
        "complete_hypothesis_retention": result["summary"]["true_hypothesis_retention"] == gates["requiredTrueHypothesisRetention"],
        "zero_individual_record_emission": result["summary"]["individual_record_emission_count"] == 0,
        "zero_execution": result["summary"]["actual_execution_count"] == 0,
    }


def main() -> None:
    lock_path = PROJECT_ROOT / "configs/v127-sgd-typed-constraint-feasibility-lock.json"
    result_path = PROJECT_ROOT / "outputs/v127-sgd-typed-constraint-feasibility/evaluation/result.json"
    doc_path = PROJECT_ROOT / "docs/v127-sgd-typed-constraint-feasibility-results.md"
    audit_path = PROJECT_ROOT / "outputs/v127-sgd-typed-constraint-feasibility/outcome-audit.json"
    outcome_path = PROJECT_ROOT / "configs/v127-sgd-typed-constraint-feasibility-outcome-lock.json"
    verifier_path = PROJECT_ROOT / "python/verify_and_freeze_v127_sgd_typed_constraint_feasibility_outcome.py"
    if audit_path.exists() or outcome_path.exists(): raise RuntimeError("V127 outcome already frozen")
    if not doc_path.is_file(): raise RuntimeError("write V127 result document first")
    lock = json.loads(lock_path.read_text()); result = json.loads(result_path.read_text())
    dependencies = [key for key in lock if not key.endswith("_sha256") and f"{key}_sha256" in lock]
    independent = aggregate_checks(result, lock["config_payload"])
    access = result["access"]
    checks = {
        "lock_and_dependencies_exact": payload_hash({key: value for key, value in lock.items() if key != "lock_payload_sha256"}) == lock["lock_payload_sha256"] and all(file_sha256(PROJECT_ROOT / lock[key]) == lock[f"{key}_sha256"] for key in dependencies),
        "aggregate_gate_classification_exact": independent == result["summary"]["outcome_gates"] and result["passed"] == all(independent.values()),
        "deterministic_recomputation_recorded": result["deterministic_in_memory_recomputation_exact"],
        "oracle_boundary_and_zero_language_value_model_execution": access["source_archive_read_count"] == 1 and access["automatic_selected_annotation_parse_count"] == 1 and all(value == 0 for key, value in access.items() if key not in {"source_archive_read_count", "automatic_selected_annotation_parse_count"}) and lock["config_payload"]["authorityBoundary"]["annotationsAreUnavailableOracleEvidenceNotRuntimeInputs"] and result["summary"]["actual_execution_count"] == 0,
    }
    passed = all(checks.values())
    audit = {"schema_version": "127-sgd-typed-constraint-feasibility-outcome-audit", "experiment": lock["config_payload"]["experiment"], "passed": passed, "checks": checks, "independent_aggregate_gate_checks": independent}
    write_json(audit_path, audit)
    if not passed: raise SystemExit(1)
    experimental_pass = bool(result["passed"])
    paths = {"analysis_lock": lock_path, "result": result_path, "verifier": verifier_path, "audit": audit_path, "results_document": doc_path}
    outcome: dict[str, Any] = {
        "schema_version": "127-sgd-typed-constraint-feasibility-outcome-lock",
        "experiment": "v127_sgd_typed_constraint_feasibility_outcome_lock",
        "outcome": {"passed": True, "audit_pass": True, "experimental_pass": experimental_pass, "decision": result["decision"], "summary": result["summary"]},
        "authorization": {
            "modify_rerun_retune_or_mine_V127": False,
            "preregister_typed_parser_realization_design": experimental_pass,
            "close_typed_constraint_signature_mechanism": not experimental_pass,
            "run_language_model_or_protected": False,
            "begin_induction_or_richer_planning": False,
            "run_API_training_action_or_execution": False,
        },
    }
    for key, path in paths.items():
        outcome[key] = str(path.relative_to(PROJECT_ROOT)); outcome[f"{key}_sha256"] = file_sha256(path)
    outcome["lock_payload_sha256"] = payload_hash(outcome)
    write_json(outcome_path, outcome)
    print(json.dumps(audit, indent=2, sort_keys=True))
    print(json.dumps({"lock": str(outcome_path.relative_to(PROJECT_ROOT)), "sha256": file_sha256(outcome_path)}, indent=2))


if __name__ == "__main__":
    main()
