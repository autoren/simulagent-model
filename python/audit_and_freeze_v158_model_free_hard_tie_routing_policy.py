#!/usr/bin/env python3
from __future__ import annotations

import json
from collections import Counter
from typing import Any

from audit_and_freeze_v122_prequery_signal_inventory import payload_hash, valid_lock, write_json
from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT


REQUEST_STAGES = {
    "request_lexical_control", "request_uncatalogued_paraphrase",
    "request_relational_tie", "request_insufficient",
}
ALLOWED_METADATA_FIELDS = {
    "fixture_id", "split", "group_id", "family_id", "stage", "stratum",
    "truth_state_id", "oracle_specific_query_id", "oracle_initial_query_id",
    "route_target_query_id", "closed_answer_event",
}


def main() -> None:
    config_path = PROJECT_ROOT / "configs/v158-model-free-hard-tie-routing-policy.json"
    plan_path = PROJECT_ROOT / "docs/v158-model-free-hard-tie-routing-policy-plan.md"
    protocol_path = PROJECT_ROOT / "python/v158_model_free_hard_tie_routing_policy.py"
    tests_path = PROJECT_ROOT / "python/test_v158_model_free_hard_tie_routing_policy.py"
    auditor_path = PROJECT_ROOT / "python/audit_and_freeze_v158_model_free_hard_tie_routing_policy.py"
    runner_path = PROJECT_ROOT / "python/run_v158_model_free_hard_tie_routing_policy.py"
    verifier_path = PROJECT_ROOT / "python/verify_and_freeze_v158_model_free_hard_tie_routing_policy_outcome.py"
    prereg_dir = PROJECT_ROOT / "outputs/v158-model-free-hard-tie-routing-policy/preregistration"
    audit_path = PROJECT_ROOT / "outputs/v158-model-free-hard-tie-routing-policy/design-audit.json"
    lock_path = PROJECT_ROOT / "configs/v158-model-free-hard-tie-routing-policy-lock.json"
    if prereg_dir.exists() or audit_path.exists() or lock_path.exists():
        raise RuntimeError("V158 already preregistered")

    config = json.loads(config_path.read_text())
    parent_path = PROJECT_ROOT / config["parentV157OutcomeLock"]
    parent = json.loads(parent_path.read_text())
    public_path = PROJECT_ROOT / parent["public_fixtures"]
    hidden_path = PROJECT_ROOT / parent["hidden_fixtures"]
    catalog_path = PROJECT_ROOT / parent["interaction_catalog"]
    witness_config_path = PROJECT_ROOT / parent["config"]
    full_public = json.loads(public_path.read_text())
    full_hidden = json.loads(hidden_path.read_text())
    full_catalog = json.loads(catalog_path.read_text())
    hidden_by_id = {row["fixture_id"]: row for row in full_hidden}
    public_requests = sorted(
        (row for row in full_public if row["split"] == "development" and hidden_by_id[row["fixture_id"]]["stage"] in REQUEST_STAGES),
        key=lambda row: row["fixture_id"],
    )
    metadata = sorted(
        ({key: row[key] for key in ALLOWED_METADATA_FIELDS} for row in full_hidden if row["split"] == "development"),
        key=lambda row: row["fixture_id"],
    )
    specific_queries = [row for row in full_catalog["queries"] if row["query_kind"] == "SPECIFIC_WITNESS"]
    retrieval_catalog = {
        "schema_version": "158-state-free-specific-retrieval-catalog-projection",
        "queries": [
            {
                "query_id": query["query_id"], "title": query["title"],
                "question": query["question"], "retrieval_profile": query["retrieval_profile"],
                "options": [{"option_id": option["option_id"], "text": option["text"]} for option in query["options"]],
            }
            for query in specific_queries
        ],
    }
    stage_counts = Counter(row["stage"] for row in metadata)
    forbidden_catalog = {"state_id", "choice_id", "witness", "truth_state_id", "oracle_query_id"}
    checks = {
        "V157_exact_and_authorizes_only_separate_model_free_development_policy": bool(
            valid_lock(parent) and parent["outcome"]["passed"]
            and parent["outcome"]["fresh_hard_tie_routing_population_pass"]
            and parent["authorization"]["preregister_model_free_development_routing_policy"]
            and not parent["authorization"]["score_policy_or_read_development_truth_before_separate_lock"]
            and not parent["authorization"]["open_or_score_evaluation_language_or_metadata"]
            and not parent["authorization"]["run_model_or_hybrid_before_model_free_routing_feasibility"]
            and not parent["authorization"]["run_API_training_induction_authority_action_or_execution"]
        ),
        "development_projections_exact": bool(
            len(public_requests) == 96 and len(metadata) == 192
            and len({row["group_id"] for row in metadata}) == 24
            and all(count == 24 for count in stage_counts.values())
            and all(row["split"] == "development" for row in public_requests + metadata)
            and all(set(row) == ALLOWED_METADATA_FIELDS and "conversation" not in row for row in metadata)
        ),
        "specific_retrieval_catalog_contains_no_state_choice_witness_truth_or_oracle_fields": bool(
            len(retrieval_catalog["queries"]) == 6
            and not any(
                forbidden_catalog & set(node)
                for query in retrieval_catalog["queries"]
                for node in [query, *query["options"], query["retrieval_profile"]]
            )
        ),
        "fixed_margin_gate_costs_and_noncompensatory_thresholds": bool(
            config["retrieval"]["minimumTopScoreForSpecific"] == 6.0
            and config["retrieval"]["minimumTopTwoMarginForSpecific"] == 4.0
            and config["retrieval"]["requireUniqueTopScore"]
            and config["retrieval"]["fallbackToGenericAfterWrongSpecific"]
            and not config["retrieval"]["stateChoiceWitnessTruthOrOracleFieldsAvailableToPolicy"]
            and not config["retrieval"]["fittingLearnedParametersOrCalibration"]
            and config["policy"]["specificQueryCost"] == 0.3
            and config["policy"]["genericRouteQueryCost"] == 0.2
            and config["gates"]["requiredMarginRouterInitialActionAccuracy"] == 1.0
            and config["gates"]["maximumMarginRouterMeanDecisionCost"] == 0.4
        ),
        "zero_policy_scoring_evaluation_model_API_training_execution_before_lock": bool(
            config["population"]["evaluationPolicyReadCount"] == 0
            and all(config["gates"][key] == 0 for key in (
                "maximumEvaluationPolicyReadCount", "maximumModelLoadCount",
                "maximumModelGenerationOrScoreCount", "maximumAPICallCount",
                "maximumTrainingRunCount", "maximumActualExecutionCount",
            ))
        ),
        "authorization_remains_fail_closed": bool(
            config["decisionRule"]["passingAuthorizesOnlySeparateLocalTieBreakerProtocolDesign"]
            and config["decisionRule"]["passingDoesNotAuthorizeImmediateModelRunV157EvaluationThresholdFitCalibrationInductionAuthorityActionOrExecution"]
        ),
        "required_files_exist": all(path.is_file() for path in (
            config_path, plan_path, protocol_path, tests_path, auditor_path, runner_path,
            verifier_path, parent_path, public_path, hidden_path, catalog_path, witness_config_path,
        )),
    }
    passed = all(checks.values())
    prereg_dir.mkdir(parents=True, exist_ok=False)
    public_projection_path = prereg_dir / "development-request-public-fixtures.json"
    metadata_projection_path = prereg_dir / "development-metadata.json"
    retrieval_catalog_path = prereg_dir / "state-free-specific-retrieval-catalog.json"
    write_json(public_projection_path, public_requests)
    write_json(metadata_projection_path, metadata)
    write_json(retrieval_catalog_path, retrieval_catalog)
    audit = {
        "schema_version": "158-model-free-hard-tie-routing-policy-design-audit",
        "experiment": config["experiment"], "passed": passed, "checks": checks,
        "policy_score_count": 0, "evaluation_policy_read_count": 0,
        "model_load_count": 0, "model_generation_or_score_count": 0,
        "API_call_count": 0, "training_run_count": 0, "actual_execution_count": 0,
    }
    write_json(audit_path, audit)
    if not passed:
        print(json.dumps(audit, indent=2, sort_keys=True))
        raise SystemExit(1)
    paths = {
        "config": config_path, "parent_outcome": parent_path, "plan": plan_path,
        "protocol": protocol_path, "tests": tests_path, "auditor": auditor_path,
        "runner": runner_path, "verifier": verifier_path, "design_audit": audit_path,
        "development_public_projection": public_projection_path,
        "development_metadata_projection": metadata_projection_path,
        "retrieval_catalog_projection": retrieval_catalog_path,
        "witness_catalog": catalog_path, "witness_config": witness_config_path,
    }
    lock: dict[str, Any] = {
        "schema_version": "158-model-free-hard-tie-routing-policy-lock",
        "experiment": config["experiment"], "config_payload": config,
        "authorization": {
            "run_single_model_free_development_policy": True,
            "read_or_score_V157_evaluation": False,
            "change_terms_weights_thresholds_costs_gates_or_comparators": False,
            "run_model_hybrid_API_training_induction_authority_action_or_execution": False,
        },
    }
    for key, path in paths.items():
        lock[key] = str(path.relative_to(PROJECT_ROOT))
        lock[f"{key}_sha256"] = file_sha256(path)
    lock["lock_payload_sha256"] = payload_hash(lock)
    write_json(lock_path, lock)
    print(json.dumps(audit, indent=2, sort_keys=True))
    print(json.dumps({"lock": str(lock_path.relative_to(PROJECT_ROOT)), "sha256": file_sha256(lock_path)}, indent=2))


if __name__ == "__main__":
    main()
