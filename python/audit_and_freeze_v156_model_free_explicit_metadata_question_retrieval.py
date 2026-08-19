#!/usr/bin/env python3
from __future__ import annotations

import json
from collections import Counter
from typing import Any

from audit_and_freeze_v122_prequery_signal_inventory import payload_hash, valid_lock, write_json
from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT


REQUEST_STAGES = {
    "request_left_anchor", "request_left_paraphrase", "request_right_clear", "request_ambiguous"
}
ALLOWED_METADATA_FIELDS = {
    "fixture_id", "split", "group_id", "family_id", "stage", "truth_state_id",
    "oracle_query_id", "closed_answer_event",
}


def main() -> None:
    config_path = PROJECT_ROOT / "configs/v156-model-free-explicit-metadata-question-retrieval.json"
    plan_path = PROJECT_ROOT / "docs/v156-model-free-explicit-metadata-question-retrieval-plan.md"
    protocol_path = PROJECT_ROOT / "python/v156_model_free_explicit_metadata_question_retrieval.py"
    tests_path = PROJECT_ROOT / "python/test_v156_model_free_explicit_metadata_question_retrieval.py"
    auditor_path = PROJECT_ROOT / "python/audit_and_freeze_v156_model_free_explicit_metadata_question_retrieval.py"
    runner_path = PROJECT_ROOT / "python/run_v156_model_free_explicit_metadata_question_retrieval.py"
    verifier_path = PROJECT_ROOT / "python/verify_and_freeze_v156_model_free_explicit_metadata_question_retrieval_outcome.py"
    prereg_dir = PROJECT_ROOT / "outputs/v156-model-free-explicit-metadata-question-retrieval/preregistration"
    audit_path = PROJECT_ROOT / "outputs/v156-model-free-explicit-metadata-question-retrieval/design-audit.json"
    lock_path = PROJECT_ROOT / "configs/v156-model-free-explicit-metadata-question-retrieval-lock.json"
    if prereg_dir.exists() or audit_path.exists() or lock_path.exists():
        raise RuntimeError("V156 already preregistered")

    config = json.loads(config_path.read_text())
    parent_path = PROJECT_ROOT / config["parentV155OutcomeLock"]
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
        (
            row for row in full_public
            if row["split"] == "development"
            and hidden_by_id[row["fixture_id"]]["stage"] in REQUEST_STAGES
        ),
        key=lambda row: row["fixture_id"],
    )
    development_metadata = sorted(
        ({key: row[key] for key in ALLOWED_METADATA_FIELDS} for row in full_hidden if row["split"] == "development"),
        key=lambda row: row["fixture_id"],
    )
    retrieval_catalog = {
        "schema_version": "156-state-free-retrieval-catalog-projection",
        "queries": [
            {
                "query_id": query["query_id"], "title": query["title"],
                "question": query["question"], "retrieval_profile": query["retrieval_profile"],
                "options": [
                    {"option_id": option["option_id"], "text": option["text"]}
                    for option in query["options"]
                ],
            }
            for query in full_catalog["queries"]
        ],
    }
    stage_counts = Counter(row["stage"] for row in development_metadata)
    forbidden_catalog = {"state_id", "choice_id", "witness", "truth_state_id", "oracle_query_id"}
    checks = {
        "V155_exact_and_authorizes_only_separate_model_free_development_preregistration": bool(
            valid_lock(parent) and parent["outcome"]["passed"]
            and parent["outcome"]["fresh_explicit_metadata_retrieval_population_pass"]
            and parent["authorization"]["preregister_model_free_development_retrieval_policy"]
            and not parent["authorization"]["score_policy_or_read_development_truth_before_separate_lock"]
            and not parent["authorization"]["open_or_score_evaluation_language_or_metadata"]
            and not parent["authorization"]["run_model_or_hybrid_before_model_free_feasibility"]
            and not parent["authorization"]["propose_or_prune_candidate_states"]
            and not parent["authorization"]["run_API_training_induction_authority_action_or_execution"]
        ),
        "development_projections_exact_without_evaluation_or_language_truth_co_location": bool(
            len(public_requests) == 96 and len(development_metadata) == 144
            and len({row["group_id"] for row in development_metadata}) == 24
            and all(count == 24 for count in stage_counts.values())
            and all(set(row) <= {"fixture_id", "split", "conversation", "closed_answer_event"} for row in public_requests)
            and all(set(row) == ALLOWED_METADATA_FIELDS and "conversation" not in row for row in development_metadata)
            and all(row["split"] == "development" for row in public_requests + development_metadata)
        ),
        "retrieval_catalog_projection_contains_no_state_choice_witness_truth_or_oracle_fields": bool(
            len(retrieval_catalog["queries"]) == 6
            and not any(
                forbidden_catalog & set(node)
                for query in retrieval_catalog["queries"]
                for node in [query, *query["options"], query["retrieval_profile"]]
            )
        ),
        "fixed_unlearned_retrieval_contract_and_noncompensatory_gates": bool(
            config["retrieval"]["anchorPhraseWeight"] == 8.0
            and config["retrieval"]["primaryTermWeight"] == 3.0
            and config["retrieval"]["secondaryTermWeight"] == 1.0
            and config["retrieval"]["visibleQuestionSurfaceTokenWeight"] == 0.25
            and config["retrieval"]["sourceOrderTieBreak"]
            and not config["retrieval"]["stateChoiceWitnessOrTruthFieldsAvailableToPolicy"]
            and not config["retrieval"]["fittingOrLearnedParameters"]
            and config["gates"]["minimumRetrievalQueryTop1Accuracy"] == 0.95
            and config["gates"]["minimumRetrievalQueryMeanReciprocalRank"] == 0.97
            and config["gates"]["maximumRetrievalMeanCorrectQueryRank"] == 1.1
            and config["gates"]["maximumRetrievalSequentialMeanDecisionCost"] == 0.33
        ),
        "zero_policy_scoring_model_evaluation_API_training_execution_before_lock": bool(
            config["population"]["evaluationPolicyReadCount"] == 0
            and config["gates"]["maximumModelLoadCount"] == 0
            and config["gates"]["maximumModelGenerationOrScoreCount"] == 0
            and config["gates"]["maximumAPICallCount"] == 0
            and config["gates"]["maximumTrainingRunCount"] == 0
            and config["gates"]["maximumActualExecutionCount"] == 0
        ),
        "authorization_remains_fail_closed": bool(
            config["decisionRule"]["passingAuthorizesOnlyFreshHardTiePopulationDesign"]
            and config["decisionRule"]["passingDoesNotAuthorizeV155EvaluationModelOrHybridRunThresholdFitCalibrationInductionAuthorityActionOrExecution"]
        ),
        "required_files_exist": all(
            path.is_file() for path in (
                config_path, plan_path, protocol_path, tests_path, auditor_path, runner_path,
                verifier_path, parent_path, public_path, hidden_path, catalog_path, witness_config_path,
            )
        ),
    }
    passed = all(checks.values())
    prereg_dir.mkdir(parents=True, exist_ok=False)
    public_projection_path = prereg_dir / "development-request-public-fixtures.json"
    metadata_projection_path = prereg_dir / "development-metadata.json"
    retrieval_catalog_path = prereg_dir / "state-free-retrieval-catalog.json"
    write_json(public_projection_path, public_requests)
    write_json(metadata_projection_path, development_metadata)
    write_json(retrieval_catalog_path, retrieval_catalog)
    audit = {
        "schema_version": "156-model-free-explicit-metadata-question-retrieval-design-audit",
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
        "schema_version": "156-model-free-explicit-metadata-question-retrieval-lock",
        "experiment": config["experiment"], "config_payload": config,
        "authorization": {
            "run_single_model_free_development_policy": True,
            "read_or_score_V155_evaluation": False,
            "change_retrieval_terms_weights_ties_gates_or_comparators": False,
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
