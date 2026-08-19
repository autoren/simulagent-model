#!/usr/bin/env python3
from __future__ import annotations

import json
from typing import Any

from audit_and_freeze_v122_prequery_signal_inventory import payload_hash, valid_lock, write_json
from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT
from v150_oracle_closed_interaction_policy import evaluate


def main() -> None:
    config_path = PROJECT_ROOT / "configs/v150-oracle-closed-interaction-policy.json"
    plan_path = PROJECT_ROOT / "docs/v150-oracle-closed-interaction-policy-plan.md"
    results_path = PROJECT_ROOT / "docs/v150-oracle-closed-interaction-policy-results.md"
    protocol_path = PROJECT_ROOT / "python/v150_oracle_closed_interaction_policy.py"
    tests_path = PROJECT_ROOT / "python/test_v150_oracle_closed_interaction_policy.py"
    runner_path = PROJECT_ROOT / "python/run_and_freeze_v150_oracle_closed_interaction_policy.py"
    audit_path = PROJECT_ROOT / "outputs/v150-oracle-closed-interaction-policy/audit.json"
    outcome_path = PROJECT_ROOT / "configs/v150-oracle-closed-interaction-policy-outcome-lock.json"
    if audit_path.exists() or outcome_path.exists():
        raise RuntimeError("V150 already frozen")

    config = json.loads(config_path.read_text())
    parent_path = PROJECT_ROOT / config["parentV149OutcomeLock"]
    parent = json.loads(parent_path.read_text())
    witness_config_path = PROJECT_ROOT / parent["config"]
    witness_config = json.loads(witness_config_path.read_text())
    catalog_path = PROJECT_ROOT / parent["interaction_catalog"]
    catalog = json.loads(catalog_path.read_text())
    hidden_path = PROJECT_ROOT / parent["hidden_fixtures"]
    hidden = json.loads(hidden_path.read_text())
    allowed_metadata = {
        "fixture_id", "split", "group_id", "family_id", "stage", "truth_state_id",
        "compatible_state_ids", "oracle_query_id", "closed_answer_event",
        "presented_candidate_choice_id",
    }
    development_metadata = [
        {key: row[key] for key in allowed_metadata}
        for row in hidden
        if row["split"] == config["population"]["split"]
    ]
    result = evaluate(development_metadata, catalog, witness_config, config)
    checks = {
        "V149_valid_and_authorizes_model_free_oracle_only": bool(
            valid_lock(parent)
            and parent["outcome"]["passed"]
            and parent["outcome"]["fresh_closed_interaction_population_pass"]
            and parent["authorization"]["run_model_free_oracle_interaction_policy"]
            and not parent["authorization"]["run_language_or_model_before_separate_preregistration"]
            and not parent["authorization"]["open_evaluation_language_during_oracle_policy"]
        ),
        "development_metadata_projection_contains_no_conversation": bool(
            len(development_metadata) == 144
            and all("conversation" not in row for row in development_metadata)
        ),
        "all_oracle_policy_value_invariance_gates_pass": result["passed"],
        "zero_evaluation_language_model_API_training_execution": all(
            result["metrics"][key] == 0
            for key in (
                "evaluation_language_read_count",
                "model_load_count",
                "model_generation_or_score_count",
                "API_call_count",
                "training_run_count",
                "actual_execution_count",
            )
        ),
        "required_files_exist": all(
            path.is_file()
            for path in (config_path, plan_path, results_path, protocol_path, tests_path, runner_path)
        ),
    }
    passed = all(checks.values())
    decision = (
        config["decisionRule"]["ifEveryOraclePolicyValueInvarianceAndAccessGatePasses"]
        if passed
        else config["decisionRule"]["otherwise"]
    )
    audit = {
        "schema_version": "150-oracle-closed-interaction-policy-audit",
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
        "schema_version": "150-oracle-closed-interaction-policy-outcome-lock",
        "experiment": config["experiment"],
        "outcome": {
            "passed": True,
            "oracle_closed_interaction_policy_feasible": True,
            "language_proposal_and_query_ranking_remain_unproven": True,
            "decision": decision,
            "metrics": result["metrics"],
            "family_metrics": result["family_metrics"],
        },
        "authorization": {
            "design_local_development_proposal_protocol": True,
            "run_language_or_model_before_separate_preregistration": False,
            "open_or_score_V149_evaluation_split": False,
            "fit_calibration_or_mine_future_results": False,
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
