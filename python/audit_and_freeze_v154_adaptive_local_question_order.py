#!/usr/bin/env python3
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer

from audit_and_freeze_v122_prequery_signal_inventory import payload_hash, valid_lock, write_json
from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT
from v154_adaptive_local_question_order import evaluate_condition, render_prompt


REQUEST_STAGES = {
    "request_known_familiar", "request_known_unfamiliar", "request_right", "request_ambiguous"
}
ANSWER_STAGES = {"closed_answer_known", "closed_answer_right"}


def _oracle_completed(hidden: list[dict[str, Any]], catalog: dict[str, Any]) -> dict[str, dict[str, Any]]:
    query_ids = [row["query_id"] for row in catalog["queries"]]
    completed = {}
    for row in hidden:
        ranking = [row["oracle_query_id"]] + [query for query in query_ids if query != row["oracle_query_id"]]
        completed[row["fixture_id"]] = {
            "ranking_valid": True,
            "validation_reason": "valid_registered_query_ranking",
            "normalized_ranking": {"query_ranking": ranking},
            "query_ranking": ranking,
            "permanently_non_authoritative": True,
            "authoritative_hypothesis_universe_pruned": False,
            "capability_defined_or_registered": False,
            "executable": False,
            "actual_execution_count": 0,
            "generated_token_count": 1,
            "generation_seconds": 0.0,
        }
    return completed


def main() -> None:
    config_path = PROJECT_ROOT / "configs/v154-adaptive-local-question-order.json"
    plan_path = PROJECT_ROOT / "docs/v154-adaptive-local-question-order-plan.md"
    protocol_path = PROJECT_ROOT / "python/v154_adaptive_local_question_order.py"
    tests_path = PROJECT_ROOT / "python/test_v154_adaptive_local_question_order.py"
    auditor_path = PROJECT_ROOT / "python/audit_and_freeze_v154_adaptive_local_question_order.py"
    runner_path = PROJECT_ROOT / "python/run_v154_adaptive_local_question_order.py"
    verifier_path = PROJECT_ROOT / "python/verify_and_freeze_v154_adaptive_local_question_order_outcome.py"
    preregistration_dir = PROJECT_ROOT / "outputs/v154-adaptive-local-question-order/preregistration"
    audit_path = PROJECT_ROOT / "outputs/v154-adaptive-local-question-order/design-audit.json"
    lock_path = PROJECT_ROOT / "configs/v154-adaptive-local-question-order-lock.json"
    if preregistration_dir.exists() or audit_path.exists() or lock_path.exists():
        raise RuntimeError("V154 already preregistered")

    config = json.loads(config_path.read_text())
    parent_path = PROJECT_ROOT / config["parentV153OutcomeLock"]
    parent = json.loads(parent_path.read_text())
    v152_path = PROJECT_ROOT / parent["parent_outcome"]
    v152 = json.loads(v152_path.read_text())
    witness_config_path = PROJECT_ROOT / parent["witness_config"]
    witness_config = json.loads(witness_config_path.read_text())
    comparator_config_path = PROJECT_ROOT / parent["config"]
    comparator_config = json.loads(comparator_config_path.read_text())
    catalog_path = PROJECT_ROOT / parent["interaction_catalog"]
    catalog = json.loads(catalog_path.read_text())
    source_public_path = PROJECT_ROOT / v152["public_fixtures"]
    source_hidden_path = PROJECT_ROOT / v152["hidden_fixtures"]
    source_public = json.loads(source_public_path.read_text())
    source_hidden = json.loads(source_hidden_path.read_text())
    public_by_id = {row["fixture_id"]: row for row in source_public}

    hidden_fields = {"fixture_id", "split", "group_id", "family_id", "stage", "oracle_query_id"}
    hidden_requests = sorted(
        (
            {key: row[key] for key in hidden_fields}
            for row in source_hidden
            if row["split"] == config["population"]["split"] and row["stage"] in REQUEST_STAGES
        ),
        key=lambda row: row["fixture_id"],
    )
    public_requests = [public_by_id[row["fixture_id"]] for row in hidden_requests]
    answer_fields = {
        "fixture_id", "split", "group_id", "family_id", "stage",
        "truth_state_id", "oracle_query_id", "closed_answer_event",
    }
    answer_metadata = sorted(
        (
            {key: row[key] for key in answer_fields}
            for row in source_hidden
            if row["split"] == config["population"]["split"] and row["stage"] in ANSWER_STAGES
        ),
        key=lambda row: row["fixture_id"],
    )

    manifest_path = PROJECT_ROOT / config["modelManifest"]
    manifest = json.loads(manifest_path.read_text())
    snapshot = Path(manifest["snapshot_path"])
    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
    direct_counts = []
    low_counts = []
    prompt_surface_checks = []
    template_checks = []
    for fixture in public_requests:
        payload = render_prompt(catalog, fixture, config)
        decoded = json.loads(payload)
        prompt_surface_checks.append(
            set(decoded) == {"instruction", "registered_clarification_questions", "conversation", "response_contract"}
            and set(decoded["response_contract"]) == {"query_ranking"}
            and all(
                set(option) == {"option_id", "text"}
                for query in decoded["registered_clarification_questions"]
                for option in query["options"]
            )
            and not any(
                forbidden in payload
                for forbidden in (
                    '"choice_id"', '"state_id"', '"truth_state_id"', '"compatible_state_ids"',
                    '"oracle_query_id"', '"oracle_witness"', '"witness"', '"candidate_state_ids"',
                )
            )
        )
        messages = [
            {"role": "system", "content": config["prompt"]["system"]},
            {"role": "user", "content": payload},
        ]
        direct = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        low = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=True,
            reasoning_effort=config["conditions"]["boundedLowReasoning"]["reasoningEffort"],
        )
        direct_counts.append(len(tokenizer.encode(direct, add_special_tokens=False)))
        low_counts.append(len(tokenizer.encode(low, add_special_tokens=False)))
        template_checks.append(
            direct.endswith("<think>\n\n</think>\n\n")
            and "Reasoning effort is set to low." in low
            and "Reasoning effort is set to xhigh." not in low
            and low.endswith("<think>\n")
        )

    oracle_summary = evaluate_condition(
        _oracle_completed(hidden_requests, catalog), hidden_requests, answer_metadata,
        catalog, witness_config, comparator_config, config,
    )
    stage_counts = Counter(row["stage"] for row in hidden_requests)
    answer_stage_counts = Counter(row["stage"] for row in answer_metadata)
    qualification = config["qualificationGates"]
    access = config["accessGates"]
    low = config["conditions"]["boundedLowReasoning"]
    checks = {
        "V153_valid_and_authorizes_local_question_order_design_only": bool(
            valid_lock(parent)
            and parent["outcome"]["passed"]
            and parent["outcome"]["question_order_comparators_feasible"]
            and parent["outcome"]["candidate_proposal_field_count"] == 0
            and parent["authorization"]["design_local_development_question_order_protocol"]
            and not parent["authorization"]["run_language_or_model_before_separate_preregistration"]
            and not parent["authorization"]["open_or_score_V152_evaluation_split"]
            and not parent["authorization"]["add_candidate_state_proposal_or_pruning"]
        ),
        "V152_source_lock_valid": bool(
            valid_lock(v152)
            and v152["outcome"]["passed"]
            and v152["outcome"]["fresh_question_order_only_population_pass"]
            and v152["outcome"]["candidate_proposal_field_count"] == 0
        ),
        "development_request_population_exact_and_aligned": bool(
            len(public_requests) == config["population"]["requestFixtureCount"]
            and len(hidden_requests) == config["population"]["requestFixtureCount"]
            and [row["fixture_id"] for row in public_requests] == [row["fixture_id"] for row in hidden_requests]
            and len({row["group_id"] for row in hidden_requests}) == config["population"]["groupCount"]
            and stage_counts == Counter({stage: 24 for stage in REQUEST_STAGES})
            and all(row["closed_answer_event"] is None for row in public_requests)
        ),
        "development_hidden_and_answer_metadata_contain_no_language_or_candidate_fields": bool(
            len(answer_metadata) == 48
            and answer_stage_counts == Counter({stage: 24 for stage in ANSWER_STAGES})
            and all(set(row) == hidden_fields for row in hidden_requests)
            and all(set(row) == answer_fields for row in answer_metadata)
            and all("conversation" not in row for row in hidden_requests + answer_metadata)
        ),
        "public_requests_hide_ground_truth": all(
            not ({"group_id", "family_id", "stage", "truth_state_id", "oracle_query_id", "oracle_witness"} & set(row))
            for row in public_requests
        ),
        "pinned_local_manifest_and_chat_template_exact": bool(
            manifest["repository"] == config["model"]["repository"]
            and manifest["revision"] == config["model"]["revision"]
            and manifest["quantization_bits"] == config["model"]["quantizationBits"]
            and snapshot.is_dir()
            and file_sha256(snapshot / "chat_template.jinja")
            == next(row["sha256"] for row in manifest["files"] if row["path"] == "chat_template.jinja")
        ),
        "question_only_prompts_and_actual_template_controls_fit_budget": bool(
            prompt_surface_checks and all(prompt_surface_checks)
            and template_checks and all(template_checks)
            and max(direct_counts + low_counts) <= config["prompt"]["maximumPromptTokens"]
        ),
        "adaptive_direct_then_bounded_low_contract_exact": bool(
            config["conditions"]["direct"]["enableThinking"] is False
            and config["conditions"]["direct"]["generationCallsPerFixture"] == 1
            and low["enableThinking"] is True
            and low["reasoningEffort"] == "low"
            and low["generationCallsPerFixture"] == 2
            and low["mechanicallyForceCloseThinkingBeforeFinalPhase"]
            and low["runOnlyIfDirectFailsQualification"]
            and config["decisionRule"]["directPassStopsBeforeBoundedLowReasoning"]
            and config["model"]["temperature"] == 0.0
            and config["model"]["samplesPerPrompt"] == 1
            and config["model"]["retryCount"] == 0
            and config["model"]["modelLoadLimit"] == 1
        ),
        "oracle_mock_passes_all_question_order_gates": bool(
            oracle_summary["qualified"] and all(oracle_summary["qualification_gates"].values())
        ),
        "noncompensatory_query_cost_and_safety_gates_present": all(
            key in qualification
            for key in (
                "minimumStructuralValidity", "minimumQueryTop1Accuracy",
                "minimumQueryMeanReciprocalRank", "maximumMeanCorrectQueryRank",
                "maximumSequentialMeanDecisionCost", "minimumSequentialImprovementOverNoQuery",
                "requiredFinalExactAccuracyAfterTrustedAnswer",
                "requiredIrrelevantQueryIntermediateFailClosedRate",
                "requiredAuthoritativeHypothesisRetention", "maximumCandidateProposalFieldCount",
                "maximumActualExecutionCount",
            )
        ),
        "access_and_authorization_fail_closed": bool(
            access["maximumDirectGenerationCount"] == len(public_requests)
            and access["maximumBoundedReasoningPhaseGenerationCount"] == len(public_requests)
            and access["maximumBoundedFinalPhaseGenerationCount"] == len(public_requests)
            and access["maximumTotalGenerationCount"] == 3 * len(public_requests)
            and all(
                access[key] == 0
                for key in (
                    "maximumClosedAnswerModelGenerationCount", "maximumEvaluationFixtureModelGenerationCount",
                    "maximumRetryCount", "maximumManualRawResponseInspectionCount",
                    "maximumPersistedRawResponseCount", "maximumAPICallCount",
                    "maximumTrainingRunCount", "maximumRealServiceCallCount",
                    "maximumExternalSideEffectCount", "maximumActualExecutionCount",
                )
            )
            and config["decisionRule"]["passAuthorizesOnlySeparateEvaluationPreregistration"]
            and not config["decisionRule"]["outcomeAuthorizesImmediateEvaluationRun"]
            and not config["decisionRule"]["outcomeAuthorizesRetryRerunRepromptReasoningChangeThresholdFitCalibrationAPITrainingInductionAuthorityActionOrExecution"]
        ),
        "required_preregistration_files_exist": all(
            path.is_file()
            for path in (config_path, plan_path, protocol_path, tests_path, auditor_path, runner_path, verifier_path)
        ),
    }
    passed = all(checks.values())
    preregistration_dir.mkdir(parents=True, exist_ok=False)
    development_public_path = preregistration_dir / "development-request-public-fixtures.json"
    development_hidden_path = preregistration_dir / "development-request-hidden-fixtures.json"
    answer_metadata_path = preregistration_dir / "development-closed-answer-metadata.json"
    write_json(development_public_path, public_requests)
    write_json(development_hidden_path, hidden_requests)
    write_json(answer_metadata_path, answer_metadata)
    audit = {
        "schema_version": "154-adaptive-local-question-order-design-audit",
        "experiment": config["experiment"],
        "passed": passed,
        "checks": checks,
        "request_fixture_count": len(public_requests),
        "answer_metadata_count": len(answer_metadata),
        "minimum_direct_prompt_token_count": min(direct_counts),
        "maximum_direct_prompt_token_count": max(direct_counts),
        "minimum_low_prompt_token_count": min(low_counts),
        "maximum_low_prompt_token_count": max(low_counts),
        "oracle_summary": oracle_summary,
        "model_load_count": 0,
        "model_generation_count": 0,
        "evaluation_fixture_model_generation_count": 0,
        "decision": "authorize_exact_single_adaptive_V154_development_run" if passed else "close_V154_before_model_run",
    }
    write_json(audit_path, audit)
    if not passed:
        print(json.dumps(audit, indent=2, sort_keys=True))
        raise SystemExit(1)

    dependencies = {
        "config": config_path,
        "parent_outcome": parent_path,
        "V152_outcome": v152_path,
        "witness_config": witness_config_path,
        "comparator_config": comparator_config_path,
        "model_manifest": manifest_path,
        "interaction_catalog": catalog_path,
        "source_public_fixtures": source_public_path,
        "source_hidden_fixtures": source_hidden_path,
        "development_public_fixtures": development_public_path,
        "development_hidden_fixtures": development_hidden_path,
        "development_answer_metadata": answer_metadata_path,
        "plan": plan_path,
        "protocol": protocol_path,
        "tests": tests_path,
        "auditor": auditor_path,
        "runner": runner_path,
        "verifier": verifier_path,
        "design_audit": audit_path,
    }
    lock: dict[str, Any] = {
        "schema_version": "154-adaptive-local-question-order-lock",
        "experiment": config["experiment"],
        "config_payload": config,
        "authorization": {
            "run_exact_single_adaptive_local_development_realization": True,
            "generate_on_closed_answer_or_V152_evaluation_fixtures": False,
            "persist_or_manually_inspect_raw_model_outputs": False,
            "add_candidate_state_proposal_confidence_or_pruning": False,
            "retry_rerun_reprompt_change_reasoning_budget_tune_threshold_fit_or_mine_V154": False,
            "run_API_training_induction_authority_action_or_execution": False,
        },
    }
    for key, path in dependencies.items():
        lock[key] = str(path.relative_to(PROJECT_ROOT))
        lock[f"{key}_sha256"] = file_sha256(path)
    lock["lock_payload_sha256"] = payload_hash(lock)
    write_json(lock_path, lock)
    print(json.dumps(audit, indent=2, sort_keys=True))
    print(json.dumps({"lock": str(lock_path.relative_to(PROJECT_ROOT)), "sha256": file_sha256(lock_path)}, indent=2))


if __name__ == "__main__":
    main()
