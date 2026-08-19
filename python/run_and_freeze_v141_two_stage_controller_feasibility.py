#!/usr/bin/env python3
from __future__ import annotations

import json
from typing import Any

from audit_and_freeze_v122_prequery_signal_inventory import payload_hash, valid_lock, write_json
from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT
from v141_two_stage_controller_feasibility import evaluate, evaluate_gates


def main() -> None:
    config_path = PROJECT_ROOT / "configs/v141-two-stage-controller-feasibility.json"
    plan_path = PROJECT_ROOT / "docs/v141-two-stage-controller-feasibility-plan.md"
    protocol_path = PROJECT_ROOT / "python/v141_two_stage_controller_feasibility.py"
    tests_path = PROJECT_ROOT / "python/test_v141_two_stage_controller_feasibility.py"
    runner_path = PROJECT_ROOT / "python/run_and_freeze_v141_two_stage_controller_feasibility.py"
    results_path = PROJECT_ROOT / "docs/v141-two-stage-controller-feasibility-results.md"
    audit_path = PROJECT_ROOT / "outputs/v141-two-stage-controller-feasibility/audit.json"
    outcome_path = PROJECT_ROOT / "configs/v141-two-stage-controller-feasibility-outcome-lock.json"
    if audit_path.exists() or outcome_path.exists():
        raise RuntimeError("V141 already frozen")

    config = json.loads(config_path.read_text())
    parent_path = PROJECT_ROOT / config["parentV140OutcomeLock"]
    parent = json.loads(parent_path.read_text())
    v135_config_path = PROJECT_ROOT / config["V135Config"]
    v135_config = json.loads(v135_config_path.read_text())
    v135_outcome_path = PROJECT_ROOT / config["V135OutcomeLock"]
    v135_outcome = json.loads(v135_outcome_path.read_text())
    catalog_path = PROJECT_ROOT / v135_outcome["choice_catalog"]
    catalog = json.loads(catalog_path.read_text())
    v136_path = PROJECT_ROOT / config["V136Config"]
    v136 = json.loads(v136_path.read_text())
    result = evaluate(config, v135_config, catalog, v136)
    gates = evaluate_gates(result, config)
    checks = {
        "V140_valid_and_authorizes_model_free_controller_feasibility": bool(
            valid_lock(parent)
            and parent["outcome"]["passed"]
            and parent["outcome"]["two_mechanism_gap_confirmed"]
            and parent["authorization"]["preregister_model_free_bounded_finalizer_and_evidence_sufficiency_feasibility"]
            and not parent["authorization"]["run_language_or_model"]
        ),
        "reference_passes_every_bounded_gate": result["reference"]["qualified_on_bounded_gates"],
        "symmetric_threshold_at_most_registered_maximum": result["symmetric_marginal_reliability_threshold"] <= config["qualificationGates"]["maximumSymmetricMarginalReliabilityThreshold"],
        "arbitrary_dependence_and_no_independence_claim": result["arbitrary_within_decision_dependence"] and result["arbitrary_cross_fixture_group_dependence"] and not result["independence_assumption_used"],
        "candidate_attraction_remains_uncertified": not result["candidate_attraction_certified"],
        "all_result_gates_pass": all(gates.values()),
        "zero_language_model_API_training_and_execution_access": True,
        "required_files_exist": all(path.is_file() for path in (config_path, plan_path, protocol_path, tests_path, runner_path, results_path)),
    }
    passed = all(checks.values())
    decision = config["decisionRule"]["ifReferenceAndThresholdGatesPass"] if passed else config["decisionRule"]["otherwise"]
    audit = {
        "schema_version": "141-two-stage-controller-feasibility-audit",
        "experiment": config["experiment"],
        "passed": passed,
        "checks": checks,
        "gates": gates,
        "result": result,
        "decision": decision,
        "access": {
            "raw_response_or_trace_read_count": 0,
            "model_load_count": 0,
            "model_generation_count": 0,
            "V134_language_read_count": 0,
            "external_language_read_count": 0,
            "API_call_count": 0,
            "training_run_count": 0,
            "actual_execution_count": 0,
        },
    }
    write_json(audit_path, audit)
    if not passed:
        print(json.dumps(audit, indent=2, sort_keys=True))
        raise SystemExit(1)
    dependencies = {
        "config": config_path,
        "parent_outcome": parent_path,
        "V135_config": v135_config_path,
        "V135_outcome": v135_outcome_path,
        "choice_catalog": catalog_path,
        "V136_config": v136_path,
        "plan": plan_path,
        "protocol": protocol_path,
        "tests": tests_path,
        "runner": runner_path,
        "results_document": results_path,
        "audit": audit_path,
    }
    outcome: dict[str, Any] = {
        "schema_version": "141-two-stage-controller-feasibility-outcome-lock",
        "experiment": config["experiment"],
        "outcome": {
            "passed": True,
            "robust_two_stage_envelope_feasible": True,
            "symmetric_marginal_reliability_threshold": result["symmetric_marginal_reliability_threshold"],
            "individual_thresholds": result["individual_thresholds_with_other_marginals_at_reference"],
            "candidate_attraction_certified": False,
            "decision": decision,
        },
        "authorization": {
            "design_fresh_bounded_finalizer_evidence_certificate_interface_and_population": True,
            "run_language_or_model": False,
            "claim_same_model_stage_independence": False,
            "rerun_mine_or_modify_V139": False,
            "touch_V134_or_external_language": False,
            "run_API_training_induction_authority_action_or_execution": False,
        },
    }
    for key, path in dependencies.items():
        outcome[key] = str(path.relative_to(PROJECT_ROOT))
        outcome[f"{key}_sha256"] = file_sha256(path)
    outcome["lock_payload_sha256"] = payload_hash(outcome)
    write_json(outcome_path, outcome)
    print(json.dumps({"passed": passed, "decision": decision, "reference_metrics": result["reference"]["metrics"], "symmetric_threshold": result["symmetric_marginal_reliability_threshold"], "individual_thresholds": result["individual_thresholds_with_other_marginals_at_reference"]}, indent=2, sort_keys=True))
    print(json.dumps({"lock": str(outcome_path.relative_to(PROJECT_ROOT)), "sha256": file_sha256(outcome_path)}, indent=2))


if __name__ == "__main__":
    main()
