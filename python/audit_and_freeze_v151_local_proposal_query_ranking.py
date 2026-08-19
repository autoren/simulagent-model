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
from v151_local_proposal_query_ranking import evaluate, render_prompt


REQUEST_STAGES = {
    "request_known_familiar",
    "request_known_unfamiliar",
    "request_right",
    "request_ambiguous",
}
ANSWER_STAGES = {"closed_answer_known", "closed_answer_right"}


def _oracle_completed(
    hidden: list[dict[str, Any]],
    catalog: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    query_ids = [row["query_id"] for row in catalog["queries"]]
    completed = {}
    for row in hidden:
        status = "NEEDS_CLARIFICATION" if row["stage"] == "request_ambiguous" else "DECIDABLE"
        ranking = [row["oracle_query_id"]] + [query for query in query_ids if query != row["oracle_query_id"]]
        normalized = {
            "evidence_status": status,
            "candidate_state_ids": list(row["compatible_state_ids"]),
            "query_ranking": ranking,
            "confidence": 1.0,
        }
        completed[row["fixture_id"]] = {
            "proposal_valid": True,
            "validation_reason": "valid_registered_proposal",
            "normalized_proposal": normalized,
            **normalized,
            "generation_seconds": 0.0,
            "generated_token_count": 1,
            "permanently_non_authoritative": True,
            "authoritative_hypothesis_universe_pruned": False,
            "capability_defined_or_registered": False,
            "executable": False,
            "actual_execution_count": 0,
        }
    return completed


def _oracle_access(count: int) -> dict[str, Any]:
    return {
        "tokenizer_load_count": 1,
        "model_load_count": 1,
        "model_generation_count": count,
        "maximum_generation_count_per_fixture": 1,
        "closed_answer_model_generation_count": 0,
        "evaluation_fixture_model_generation_count": 0,
        "retry_count": 0,
        "manual_raw_response_inspection_count": 0,
        "persisted_raw_response_count": 0,
        "API_call_count": 0,
        "training_run_count": 0,
        "real_service_call_count": 0,
        "external_side_effect_count": 0,
        "actual_execution_count": 0,
    }


def main() -> None:
    config_path = PROJECT_ROOT / "configs/v151-local-proposal-query-ranking.json"
    plan_path = PROJECT_ROOT / "docs/v151-local-proposal-query-ranking-plan.md"
    protocol_path = PROJECT_ROOT / "python/v151_local_proposal_query_ranking.py"
    tests_path = PROJECT_ROOT / "python/test_v151_local_proposal_query_ranking.py"
    auditor_path = PROJECT_ROOT / "python/audit_and_freeze_v151_local_proposal_query_ranking.py"
    runner_path = PROJECT_ROOT / "python/run_v151_local_proposal_query_ranking.py"
    verifier_path = PROJECT_ROOT / "python/verify_and_freeze_v151_local_proposal_query_ranking_outcome.py"
    preregistration_dir = PROJECT_ROOT / "outputs/v151-local-proposal-query-ranking/preregistration"
    audit_path = PROJECT_ROOT / "outputs/v151-local-proposal-query-ranking/design-audit.json"
    lock_path = PROJECT_ROOT / "configs/v151-local-proposal-query-ranking-lock.json"
    if preregistration_dir.exists() or audit_path.exists() or lock_path.exists():
        raise RuntimeError("V151 already preregistered")

    config = json.loads(config_path.read_text())
    parent_path = PROJECT_ROOT / config["parentV150OutcomeLock"]
    parent = json.loads(parent_path.read_text())
    v149_outcome_path = PROJECT_ROOT / parent["parent_outcome"]
    v149_outcome = json.loads(v149_outcome_path.read_text())
    witness_config_path = PROJECT_ROOT / parent["witness_config"]
    witness_config = json.loads(witness_config_path.read_text())
    oracle_config_path = PROJECT_ROOT / parent["config"]
    oracle_config = json.loads(oracle_config_path.read_text())
    catalog_path = PROJECT_ROOT / parent["interaction_catalog"]
    catalog = json.loads(catalog_path.read_text())
    source_public_path = PROJECT_ROOT / v149_outcome["public_fixtures"]
    source_hidden_path = PROJECT_ROOT / v149_outcome["hidden_fixtures"]
    source_public = json.loads(source_public_path.read_text())
    source_hidden = json.loads(source_hidden_path.read_text())
    public_by_id = {row["fixture_id"]: row for row in source_public}

    hidden_requests = sorted(
        (
            row for row in source_hidden
            if row["split"] == config["population"]["split"] and row["stage"] in REQUEST_STAGES
        ),
        key=lambda row: row["fixture_id"],
    )
    public_requests = [public_by_id[row["fixture_id"]] for row in hidden_requests]
    answer_fields = {
        "fixture_id",
        "split",
        "group_id",
        "family_id",
        "stage",
        "truth_state_id",
        "oracle_query_id",
        "closed_answer_event",
        "presented_candidate_choice_id",
    }
    answer_metadata = sorted(
        (
            {key: row[key] for key in answer_fields}
            for row in source_hidden
            if row["split"] == config["population"]["split"] and row["stage"] in ANSWER_STAGES
        ),
        key=lambda row: row["fixture_id"],
    )
    public_forbidden = {
        "group_id",
        "family_id",
        "stage",
        "language_class",
        "truth_state_id",
        "compatible_state_ids",
        "oracle_query_id",
        "oracle_witness",
        "trusted_witness_available",
        "variant_index",
    }

    manifest_path = PROJECT_ROOT / config["modelManifest"]
    manifest = json.loads(manifest_path.read_text())
    snapshot = Path(manifest["snapshot_path"])
    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
    prompt_token_counts = []
    prompt_surface_checks = []
    for fixture in public_requests:
        payload = render_prompt(catalog, fixture, config)
        decoded = json.loads(payload)
        prompt_surface_checks.append(
            all(
                set(option) == {"option_id", "text"}
                for query in decoded["registered_clarification_questions"]
                for option in query["options"]
            )
            and not any(
                forbidden in payload
                for forbidden in (
                    '"truth_state_id"',
                    '"compatible_state_ids"',
                    '"oracle_query_id"',
                    '"oracle_witness"',
                    '"witness"',
                )
            )
        )
        prompt = tokenizer.apply_chat_template(
            [
                {"role": "system", "content": config["prompt"]["system"]},
                {"role": "user", "content": payload},
            ],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=config["model"]["enableThinking"],
        )
        prompt_token_counts.append(len(tokenizer.encode(prompt, add_special_tokens=False)))

    oracle_summary = evaluate(
        _oracle_completed(hidden_requests, catalog),
        hidden_requests,
        answer_metadata,
        catalog,
        witness_config,
        oracle_config,
        _oracle_access(len(public_requests)),
        config,
    )
    stage_counts = Counter(row["stage"] for row in hidden_requests)
    answer_stage_counts = Counter(row["stage"] for row in answer_metadata)
    qualification = config["qualificationGates"]
    access = config["accessGates"]
    checks = {
        "V150_valid_and_authorizes_local_protocol_design_only": bool(
            valid_lock(parent)
            and parent["outcome"]["passed"]
            and parent["outcome"]["oracle_closed_interaction_policy_feasible"]
            and parent["authorization"]["design_local_development_proposal_protocol"]
            and not parent["authorization"]["run_language_or_model_before_separate_preregistration"]
            and not parent["authorization"]["open_or_score_V149_evaluation_split"]
        ),
        "V149_source_lock_valid": bool(
            valid_lock(v149_outcome)
            and v149_outcome["outcome"]["passed"]
            and v149_outcome["outcome"]["fresh_closed_interaction_population_pass"]
        ),
        "development_request_population_exact_and_aligned": bool(
            len(public_requests) == config["population"]["requestFixtureCount"]
            and len(hidden_requests) == config["population"]["requestFixtureCount"]
            and [row["fixture_id"] for row in public_requests] == [row["fixture_id"] for row in hidden_requests]
            and len({row["group_id"] for row in hidden_requests}) == config["population"]["groupCount"]
            and stage_counts == Counter({stage: 24 for stage in REQUEST_STAGES})
            and all(row["closed_answer_event"] is None for row in public_requests)
        ),
        "development_answer_metadata_exact_and_contains_no_language": bool(
            len(answer_metadata) == 48
            and answer_stage_counts == Counter({stage: 24 for stage in ANSWER_STAGES})
            and all("conversation" not in row for row in answer_metadata)
            and all(set(row) == answer_fields for row in answer_metadata)
        ),
        "public_request_fixtures_hide_ground_truth": all(not (public_forbidden & set(row)) for row in public_requests),
        "pinned_local_model_manifest_exact_and_snapshot_exists": bool(
            manifest["repository"] == config["model"]["repository"]
            and manifest["revision"] == config["model"]["revision"]
            and manifest["quantization_bits"] == config["model"]["quantizationBits"]
            and snapshot.is_dir()
        ),
        "all_prompts_hide_typed_answer_mapping_and_fit_budget": bool(
            prompt_surface_checks
            and all(prompt_surface_checks)
            and max(prompt_token_counts) <= config["prompt"]["maximumPromptTokens"]
        ),
        "single_direct_generation_no_retry_contract": bool(
            not config["model"]["enableThinking"]
            and config["model"]["temperature"] == 0.0
            and config["model"]["samplesPerPrompt"] == 1
            and config["model"]["retryCount"] == 0
            and config["model"]["modelLoadLimit"] == 1
            and access["maximumModelGenerationCount"] == len(public_requests)
            and access["maximumGenerationCountPerFixture"] == 1
            and access["maximumClosedAnswerModelGenerationCount"] == 0
            and access["maximumEvaluationFixtureModelGenerationCount"] == 0
        ),
        "oracle_mock_passes_every_noncompensatory_gate": bool(
            oracle_summary["qualified"]
            and all(oracle_summary["qualification_gates"].values())
            and all(oracle_summary["access_gates"].values())
        ),
        "semantic_query_sequential_and_safety_gates_present": all(
            key in qualification
            for key in (
                "minimumProposalStructuralValidity",
                "minimumCompatibleStateRetention",
                "minimumEveryLanguageClassCompatibleStateRetention",
                "minimumCompatibleSetExactAccuracy",
                "minimumAmbiguitySensitivity",
                "minimumDecidableSpecificity",
                "minimumDecidableTop1Accuracy",
                "minimumQueryTop1Accuracy",
                "minimumQueryMeanReciprocalRank",
                "maximumMeanCorrectQueryRank",
                "maximumFalseKnownSingletonRateOnNonKnownTruth",
                "maximumCandidateAttractedFractionOfSemanticErrors",
                "maximumSequentialMeanDecisionCost",
                "minimumSequentialImprovementOverNoQuery",
                "requiredFinalExactAccuracyAfterTrustedAnswer",
                "requiredIrrelevantQueryIntermediateFailClosedRate",
                "requiredAuthoritativeTrueHypothesisRetention",
                "maximumActualExecutionCount",
            )
        ),
        "zero_closed_answer_evaluation_retry_raw_API_training_authority_execution": bool(
            all(
                access[key] == 0
                for key in (
                    "maximumClosedAnswerModelGenerationCount",
                    "maximumEvaluationFixtureModelGenerationCount",
                    "maximumRetryCount",
                    "maximumManualRawResponseInspectionCount",
                    "maximumPersistedRawResponseCount",
                    "maximumAPICallCount",
                    "maximumTrainingRunCount",
                    "maximumRealServiceCallCount",
                    "maximumExternalSideEffectCount",
                    "maximumActualExecutionCount",
                )
            )
            and config["decisionRule"]["passAuthorizesOnlySeparateEvaluationPreregistration"]
            and not config["decisionRule"]["outcomeAuthorizesImmediateEvaluationRun"]
            and not config["decisionRule"]["outcomeAuthorizesRetryRerunPromptChangeThresholdFitAPITrainingInductionAuthorityActionOrExecution"]
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
        "schema_version": "151-local-proposal-query-ranking-design-audit",
        "experiment": config["experiment"],
        "passed": passed,
        "checks": checks,
        "request_fixture_count": len(public_requests),
        "answer_metadata_count": len(answer_metadata),
        "minimum_prompt_token_count": min(prompt_token_counts),
        "maximum_prompt_token_count": max(prompt_token_counts),
        "oracle_summary": oracle_summary,
        "model_load_count": 0,
        "model_generation_count": 0,
        "evaluation_fixture_model_generation_count": 0,
        "decision": "authorize_exact_single_V151_development_run" if passed else "close_V151_before_model_run",
    }
    write_json(audit_path, audit)
    if not passed:
        print(json.dumps(audit, indent=2, sort_keys=True))
        raise SystemExit(1)

    dependencies = {
        "config": config_path,
        "parent_outcome": parent_path,
        "V149_outcome": v149_outcome_path,
        "witness_config": witness_config_path,
        "oracle_config": oracle_config_path,
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
        "schema_version": "151-local-proposal-query-ranking-lock",
        "experiment": config["experiment"],
        "config_payload": config,
        "authorization": {
            "run_exact_single_pinned_local_development_realization": True,
            "modify_retry_rerun_reprompt_tune_threshold_fit_or_mine_V151": False,
            "generate_on_closed_answer_or_V149_evaluation_fixtures": False,
            "persist_or_manually_inspect_raw_model_responses": False,
            "run_API_training_induction_authority_action_or_execution": False,
        },
    }
    for key, path in dependencies.items():
        lock[key] = str(path.relative_to(PROJECT_ROOT))
        lock[f"{key}_sha256"] = file_sha256(path)
    lock["lock_payload_sha256"] = payload_hash(lock)
    write_json(lock_path, lock)
    print(json.dumps({"passed": passed, "checks": checks, "decision": audit["decision"]}, indent=2, sort_keys=True))
    print(json.dumps({"lock": str(lock_path.relative_to(PROJECT_ROOT)), "sha256": file_sha256(lock_path)}, indent=2))


if __name__ == "__main__":
    main()
