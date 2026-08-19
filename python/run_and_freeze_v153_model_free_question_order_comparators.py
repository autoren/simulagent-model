#!/usr/bin/env python3
from __future__ import annotations

import json
from typing import Any

from audit_and_freeze_v122_prequery_signal_inventory import payload_hash, valid_lock, write_json
from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT
from v153_model_free_question_order_comparators import evaluate


def main() -> None:
    config_path = PROJECT_ROOT / "configs/v153-model-free-question-order-comparators.json"
    plan_path = PROJECT_ROOT / "docs/v153-model-free-question-order-comparators-plan.md"
    results_path = PROJECT_ROOT / "docs/v153-model-free-question-order-comparators-results.md"
    protocol_path = PROJECT_ROOT / "python/v153_model_free_question_order_comparators.py"
    tests_path = PROJECT_ROOT / "python/test_v153_model_free_question_order_comparators.py"
    runner_path = PROJECT_ROOT / "python/run_and_freeze_v153_model_free_question_order_comparators.py"
    audit_path = PROJECT_ROOT / "outputs/v153-model-free-question-order-comparators/audit.json"
    outcome_path = PROJECT_ROOT / "configs/v153-model-free-question-order-comparators-outcome-lock.json"
    if audit_path.exists() or outcome_path.exists():
        raise RuntimeError("V153 already frozen")

    config = json.loads(config_path.read_text())
    parent_path = PROJECT_ROOT / config["parentV152OutcomeLock"]
    parent = json.loads(parent_path.read_text())
    witness_config_path = PROJECT_ROOT / parent["config"]
    witness_config = json.loads(witness_config_path.read_text())
    catalog_path = PROJECT_ROOT / parent["interaction_catalog"]
    catalog = json.loads(catalog_path.read_text())
    hidden_path = PROJECT_ROOT / parent["hidden_fixtures"]
    hidden = json.loads(hidden_path.read_text())
    allowed_metadata = {
        "fixture_id", "split", "group_id", "family_id", "stage", "truth_state_id",
        "oracle_query_id", "closed_answer_event",
    }
    development_metadata = [
        {key: row[key] for key in allowed_metadata}
        for row in hidden
        if row["split"] == config["population"]["split"]
    ]
    result = evaluate(development_metadata, catalog, witness_config, config)
    checks = {
        "V152_exact_and_authorizes_only_model_free_comparators": bool(
            valid_lock(parent)
            and parent["outcome"]["passed"]
            and parent["outcome"]["fresh_question_order_only_population_pass"]
            and parent["outcome"]["candidate_proposal_field_count"] == 0
            and parent["authorization"]["run_model_free_question_order_comparator_policy"]
            and not parent["authorization"]["run_language_or_model_before_separate_preregistration"]
            and not parent["authorization"]["open_evaluation_language_or_metadata_during_comparator_policy"]
            and not parent["authorization"]["propose_or_prune_candidate_states"]
            and not parent["authorization"]["run_API_training_calibration_induction_authority_action_or_execution"]
        ),
        "development_projection_contains_no_conversation_or_candidate_fields": bool(
            len(development_metadata) == 144
            and all("conversation" not in row for row in development_metadata)
            and result["candidate_proposal_field_count"] == 0
        ),
        "all_comparator_firewall_and_access_gates_pass": result["passed"],
        "zero_evaluation_model_API_training_execution": all(
            result[key] == 0
            for key in (
                "evaluation_language_read_count", "model_load_count", "model_generation_or_score_count",
                "API_call_count", "training_run_count", "actual_execution_count",
            )
        ),
        "required_files_exist": all(
            path.is_file()
            for path in (config_path, plan_path, results_path, protocol_path, tests_path, runner_path)
        ),
    }
    passed = all(checks.values())
    decision = (
        config["decisionRule"]["ifEveryComparatorFirewallAndAccessGatePasses"]
        if passed
        else config["decisionRule"]["otherwise"]
    )
    audit = {
        "schema_version": "153-model-free-question-order-comparators-audit",
        "experiment": config["experiment"],
        "passed": passed,
        "checks": checks,
        "result": result,
        "decision": decision,
    }
    write_json(audit_path, audit)
    if not passed:
        print(json.dumps(audit, indent=2, sort_keys=True))
        raise SystemExit(1)

    dependencies = {
        "config": config_path,
        "parent_outcome": parent_path,
        "witness_config": witness_config_path,
        "interaction_catalog": catalog_path,
        "hidden_fixtures": hidden_path,
        "plan": plan_path,
        "results_document": results_path,
        "protocol": protocol_path,
        "tests": tests_path,
        "runner": runner_path,
        "audit": audit_path,
    }
    outcome: dict[str, Any] = {
        "schema_version": "153-model-free-question-order-comparators-outcome-lock",
        "experiment": config["experiment"],
        "outcome": {
            "passed": True,
            "question_order_comparators_feasible": True,
            "decision": decision,
            "metrics": result["metrics"],
            "episode_count": result["episode_count"],
            "candidate_proposal_field_count": result["candidate_proposal_field_count"],
        },
        "authorization": {
            "design_local_development_question_order_protocol": True,
            "run_language_or_model_before_separate_preregistration": False,
            "open_or_score_V152_evaluation_split": False,
            "add_candidate_state_proposal_or_pruning": False,
            "compare_reasoning_conditions_before_separate_preregistration": False,
            "fit_calibration_thresholds_or_mine_future_results": False,
            "run_API_training_induction_authority_action_or_execution": False,
        },
    }
    for key, path in dependencies.items():
        outcome[key] = str(path.relative_to(PROJECT_ROOT))
        outcome[f"{key}_sha256"] = file_sha256(path)
    outcome["lock_payload_sha256"] = payload_hash(outcome)
    write_json(outcome_path, outcome)
    print(json.dumps({"passed": passed, "decision": decision, "metrics": result["metrics"], "checks": checks}, indent=2, sort_keys=True))
    print(json.dumps({"lock": str(outcome_path.relative_to(PROJECT_ROOT)), "sha256": file_sha256(outcome_path)}, indent=2))


if __name__ == "__main__":
    main()
