#!/usr/bin/env python3
from __future__ import annotations

import json
from typing import Any

from audit_and_freeze_v122_prequery_signal_inventory import payload_hash, valid_lock, write_json
from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT
from v157_fresh_hard_tie_routing_population import audit_population, build_population


def main() -> None:
    config_path = PROJECT_ROOT / "configs/v157-fresh-hard-tie-routing-population.json"
    plan_path = PROJECT_ROOT / "docs/v157-fresh-hard-tie-routing-population-plan.md"
    results_path = PROJECT_ROOT / "docs/v157-fresh-hard-tie-routing-population-results.md"
    protocol_path = PROJECT_ROOT / "python/v157_fresh_hard_tie_routing_population.py"
    tests_path = PROJECT_ROOT / "python/test_v157_fresh_hard_tie_routing_population.py"
    auditor_path = PROJECT_ROOT / "python/audit_and_freeze_v157_fresh_hard_tie_routing_population.py"
    output_dir = PROJECT_ROOT / "outputs/v157-fresh-hard-tie-routing-population/design"
    audit_path = PROJECT_ROOT / "outputs/v157-fresh-hard-tie-routing-population/design-audit.json"
    outcome_path = PROJECT_ROOT / "configs/v157-fresh-hard-tie-routing-population-outcome-lock.json"
    if output_dir.exists() or audit_path.exists() or outcome_path.exists():
        raise RuntimeError("V157 already frozen")

    config = json.loads(config_path.read_text())
    parent_path = PROJECT_ROOT / config["parentV156OutcomeLock"]
    parent = json.loads(parent_path.read_text())
    prior_paths = [PROJECT_ROOT / path for path in config["priorPublicPaths"]]
    prior_rows = [row for path in prior_paths for row in json.loads(path.read_text())]
    population = build_population(config)
    result = audit_population(population, config, prior_rows)
    checks = {
        "V156_exact_positive_model_free_result_authorizes_only_fresh_hard_tie_design": bool(
            valid_lock(parent) and parent["outcome"]["passed"] and parent["outcome"]["policy_qualified"]
            and parent["authorization"]["design_fresh_hard_tie_population_if_policy_qualified"]
            and not parent["authorization"]["open_or_score_V155_evaluation"]
            and not parent["authorization"]["run_model_or_hybrid_on_V155"]
            and not parent["authorization"]["change_retrieval_terms_weights_ties_gates_or_rerun"]
            and not parent["authorization"]["run_API_training_induction_authority_action_or_execution"]
        ),
        "all_hard_tie_routing_freshness_and_firewall_gates_pass": result["passed"],
        "specific_answers_route_exactly": result["specific_answer_witness_routing"] == 1.0,
        "generic_routes_valid_without_semantic_witness": bool(
            result["generic_route_validity"] == 1.0 and result["generic_route_semantic_witness_count"] == 0
        ),
        "preanswer_generic_and_malformed_fail_closed": bool(
            result["preanswer_and_generic_final_fail_closed_rate"] == 1.0
            and result["malformed_event_fail_closed_rate"] == 1.0
        ),
        "zero_candidate_fields_and_prior_overlap": bool(
            result["candidate_proposal_field_count"] == 0
            and result["exact_prior_conversation_overlap_count"] == 0
        ),
        "zero_policy_model_API_training_execution": all(
            result[key] == 0 for key in (
                "policy_score_count", "model_load_count", "model_generation_or_score_count",
                "API_call_count", "training_run_count", "actual_execution_count",
            )
        ),
        "required_files_and_prior_assets_exist": all(
            path.is_file() for path in (
                config_path, plan_path, results_path, protocol_path, tests_path, auditor_path,
                parent_path, *prior_paths,
            )
        ),
    }
    passed = all(checks.values())
    decision = (
        config["decisionRule"]["ifEveryHardTieRoutingFreshnessFirewallAndAccessGatePasses"]
        if passed else config["decisionRule"]["otherwise"]
    )
    output_dir.mkdir(parents=True, exist_ok=False)
    assets = {
        "interaction_catalog": output_dir / "interaction-catalog.json",
        "public_fixtures": output_dir / "public-fixtures.json",
        "hidden_fixtures": output_dir / "hidden-fixtures.json",
        "population_summary": output_dir / "population-summary.json",
    }
    for key, path in assets.items():
        write_json(path, population[key])
    audit = {
        "schema_version": "157-fresh-hard-tie-routing-population-design-audit",
        "experiment": config["experiment"], "passed": passed, "checks": checks,
        "result": result, "decision": decision,
        "access": {
            "policy_score_count": 0, "model_load_count": 0,
            "model_generation_or_score_count": 0, "API_call_count": 0,
            "training_run_count": 0, "actual_execution_count": 0,
        },
    }
    write_json(audit_path, audit)
    if not passed:
        print(json.dumps(audit, indent=2, sort_keys=True))
        raise SystemExit(1)
    dependencies = {
        "config": config_path, "parent_outcome": parent_path, "plan": plan_path,
        "results_document": results_path, "protocol": protocol_path, "tests": tests_path,
        "auditor": auditor_path, "design_audit": audit_path, **assets,
    }
    outcome: dict[str, Any] = {
        "schema_version": "157-fresh-hard-tie-routing-population-outcome-lock",
        "experiment": config["experiment"],
        "outcome": {
            "passed": True, "fresh_hard_tie_routing_population_pass": True,
            "decision": decision, "summary": result["summary"],
            "specific_answer_witness_routing": 1.0, "generic_route_validity": 1.0,
            "generic_route_semantic_witness_count": 0, "candidate_proposal_field_count": 0,
        },
        "authorization": {
            "preregister_model_free_development_routing_policy": True,
            "score_policy_or_read_development_truth_before_separate_lock": False,
            "open_or_score_evaluation_language_or_metadata": False,
            "run_model_or_hybrid_before_model_free_routing_feasibility": False,
            "propose_or_prune_candidate_states": False,
            "fit_thresholds_or_calibration": False,
            "run_API_training_induction_authority_action_or_execution": False,
        },
    }
    for key, path in dependencies.items():
        outcome[key] = str(path.relative_to(PROJECT_ROOT))
        outcome[f"{key}_sha256"] = file_sha256(path)
    outcome["lock_payload_sha256"] = payload_hash(outcome)
    write_json(outcome_path, outcome)
    print(json.dumps({"passed": passed, "decision": decision, "summary": result["summary"], "checks": checks}, indent=2, sort_keys=True))
    print(json.dumps({"lock": str(outcome_path.relative_to(PROJECT_ROOT)), "sha256": file_sha256(outcome_path)}, indent=2))


if __name__ == "__main__":
    main()
