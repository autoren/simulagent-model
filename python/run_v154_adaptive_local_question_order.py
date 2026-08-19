#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time
from typing import Any

import mlx.core as mx
from mlx_lm import generate, load, stream_generate
from mlx_lm.sample_utils import make_sampler

from audit_and_freeze_v122_prequery_signal_inventory import payload_hash
from locked_census_harness import run_locked_census_once, write_json
from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT
from v154_adaptive_local_question_order import (
    evaluate_condition,
    parse_ranking,
    prepare_bounded_final_prompt_tokens,
    render_prompt,
)


def main() -> None:
    lock_path = PROJECT_ROOT / "configs/v154-adaptive-local-question-order-lock.json"
    output_dir = PROJECT_ROOT / "outputs/v154-adaptive-local-question-order/model-realization"
    if output_dir.exists():
        raise RuntimeError("V154 exact adaptive development realization may run only once")
    lock = json.loads(lock_path.read_text())
    if payload_hash({key: value for key, value in lock.items() if key != "lock_payload_sha256"}) != lock["lock_payload_sha256"]:
        raise RuntimeError("V154 lock mismatch")
    dependencies = [key for key in lock if not key.endswith("_sha256") and f"{key}_sha256" in lock]
    for key in dependencies:
        if file_sha256(PROJECT_ROOT / lock[key]) != lock[f"{key}_sha256"]:
            raise RuntimeError(f"V154 dependency drifted: {key}")
    authorization = lock["authorization"]
    if (
        not authorization["run_exact_single_adaptive_local_development_realization"]
        or authorization["generate_on_closed_answer_or_V152_evaluation_fixtures"]
        or authorization["persist_or_manually_inspect_raw_model_outputs"]
        or authorization["add_candidate_state_proposal_confidence_or_pruning"]
        or authorization["run_API_training_induction_authority_action_or_execution"]
    ):
        raise RuntimeError("V154 authorization is invalid")

    config = lock["config_payload"]
    public_rows = json.loads((PROJECT_ROOT / lock["development_public_fixtures"]).read_text())
    hidden_rows = json.loads((PROJECT_ROOT / lock["development_hidden_fixtures"]).read_text())
    answer_metadata = json.loads((PROJECT_ROOT / lock["development_answer_metadata"]).read_text())
    catalog = json.loads((PROJECT_ROOT / lock["interaction_catalog"]).read_text())
    witness_config = json.loads((PROJECT_ROOT / lock["witness_config"]).read_text())
    comparator_config = json.loads((PROJECT_ROOT / lock["comparator_config"]).read_text())
    manifest = json.loads((PROJECT_ROOT / lock["model_manifest"]).read_text())
    snapshot = Path(manifest["snapshot_path"])
    if (
        len(public_rows) != config["population"]["requestFixtureCount"]
        or any(row["split"] != "development" or row["closed_answer_event"] is not None for row in public_rows)
    ):
        raise RuntimeError("V154 locked development request population mismatch")

    fixture_rows = [{"name": row["fixture_id"], "fixture": row} for row in public_rows]
    access: dict[str, Any] = {
        "tokenizer_load_count": 0,
        "model_load_count": 0,
        "direct_generation_count": 0,
        "bounded_reasoning_phase_generation_count": 0,
        "bounded_final_phase_generation_count": 0,
        "total_generation_count": 0,
        "maximum_generation_calls_per_condition_fixture": 0,
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
    state: dict[str, Any] = {"model": None, "tokenizer": None}
    sampler = make_sampler(temp=config["model"]["temperature"])
    mx.reset_peak_memory()
    started = time.perf_counter()

    def ensure_loaded() -> None:
        if state["model"] is not None:
            return
        load_started = time.perf_counter()
        model, tokenizer = load(str(snapshot))
        model.eval()
        state["model"], state["tokenizer"] = model, tokenizer
        access["model_load_count"] += 1
        access["tokenizer_load_count"] += 1
        access["model_load_seconds"] = time.perf_counter() - load_started

    def persist_progress() -> None:
        access["total_generation_count"] = (
            access["direct_generation_count"]
            + access["bounded_reasoning_phase_generation_count"]
            + access["bounded_final_phase_generation_count"]
        )
        access["elapsed_seconds"] = time.perf_counter() - started
        access["peak_active_memory_bytes"] = int(mx.get_peak_memory())
        write_json(output_dir / "access-progress.json", access)

    def base_prompt(fixture: dict[str, Any], condition: str) -> tuple[str, list[int]]:
        payload = render_prompt(catalog, fixture, config)
        messages = [
            {"role": "system", "content": config["prompt"]["system"]},
            {"role": "user", "content": payload},
        ]
        if condition == "direct":
            prompt = state["tokenizer"].apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        else:
            prompt = state["tokenizer"].apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=True,
                reasoning_effort=config["conditions"]["boundedLowReasoning"]["reasoningEffort"],
            )
        prompt_tokens = state["tokenizer"].encode(prompt, add_special_tokens=False)
        if len(prompt_tokens) > config["prompt"]["maximumPromptTokens"]:
            raise RuntimeError("V154 prompt exceeds frozen budget")
        return prompt, list(prompt_tokens)

    def evaluate_direct(row: dict[str, Any]) -> dict[str, Any]:
        ensure_loaded()
        fixture = row["fixture"]
        prompt, prompt_tokens = base_prompt(fixture, "direct")
        access["direct_generation_count"] += 1
        access["maximum_generation_calls_per_condition_fixture"] = max(
            access["maximum_generation_calls_per_condition_fixture"], 1
        )
        persist_progress()
        generation_started = time.perf_counter()
        raw = generate(
            state["model"], state["tokenizer"], prompt=prompt,
            max_tokens=config["conditions"]["direct"]["maximumNewTokens"],
            sampler=sampler, verbose=False,
        )
        elapsed = time.perf_counter() - generation_started
        generated_tokens = state["tokenizer"].encode(raw, add_special_tokens=False)
        parsed = parse_ranking(raw, catalog, config)
        persist_progress()
        return {
            "name": row["name"],
            "fixture_id": fixture["fixture_id"],
            "condition_id": config["conditions"]["direct"]["conditionId"],
            **parsed,
            "raw_response_sha256": hashlib.sha256(raw.encode()).hexdigest(),
            "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
            "prompt_token_count": len(prompt_tokens),
            "generated_token_count": len(generated_tokens),
            "maximum_new_tokens_hit": len(generated_tokens) >= config["conditions"]["direct"]["maximumNewTokens"],
            "generation_seconds": elapsed,
            "raw_response_persisted": False,
        }

    def evaluate_bounded_low(row: dict[str, Any]) -> dict[str, Any]:
        ensure_loaded()
        fixture = row["fixture"]
        prompt, prompt_tokens = base_prompt(fixture, "bounded_low")
        low = config["conditions"]["boundedLowReasoning"]
        access["bounded_reasoning_phase_generation_count"] += 1
        access["maximum_generation_calls_per_condition_fixture"] = max(
            access["maximum_generation_calls_per_condition_fixture"], 2
        )
        persist_progress()
        generation_started = time.perf_counter()
        responses = list(
            stream_generate(
                state["model"], state["tokenizer"], prompt=prompt_tokens,
                max_tokens=low["reasoningPhaseMaximumTokens"], sampler=sampler,
            )
        )
        reasoning_tokens = [response.token for response in responses]
        reasoning_text = state["tokenizer"].decode(reasoning_tokens)
        final_prompt_tokens, natural_close, retained_count = prepare_bounded_final_prompt_tokens(
            prompt_tokens, reasoning_tokens, state["tokenizer"]
        )
        access["bounded_final_phase_generation_count"] += 1
        persist_progress()
        raw = generate(
            state["model"], state["tokenizer"], prompt=final_prompt_tokens,
            max_tokens=low["finalPhaseMaximumTokens"], sampler=sampler, verbose=False,
        )
        elapsed = time.perf_counter() - generation_started
        final_tokens = state["tokenizer"].encode(raw, add_special_tokens=False)
        parsed = parse_ranking(raw, catalog, config)
        persist_progress()
        return {
            "name": row["name"],
            "fixture_id": fixture["fixture_id"],
            "condition_id": low["conditionId"],
            **parsed,
            "reasoning_response_sha256": hashlib.sha256(reasoning_text.encode()).hexdigest(),
            "raw_response_sha256": hashlib.sha256(raw.encode()).hexdigest(),
            "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
            "bounded_final_prompt_sha256": hashlib.sha256(
                bytes(str(final_prompt_tokens), "utf-8")
            ).hexdigest(),
            "prompt_token_count": len(prompt_tokens),
            "reasoning_phase_generated_token_count": len(reasoning_tokens),
            "reasoning_phase_retained_token_count": retained_count,
            "reasoning_naturally_closed_within_budget": natural_close,
            "reasoning_phase_maximum_tokens_hit": len(reasoning_tokens) >= low["reasoningPhaseMaximumTokens"],
            "final_phase_generated_token_count": len(final_tokens),
            "final_phase_maximum_tokens_hit": len(final_tokens) >= low["finalPhaseMaximumTokens"],
            "generated_token_count": len(reasoning_tokens) + len(final_tokens),
            "generation_seconds": elapsed,
            "raw_response_persisted": False,
        }

    def run_condition(name: str, evaluator: Any) -> tuple[dict[str, Any], dict[str, Any]]:
        summary_holder: dict[str, Any] = {"summary": None}

        def aggregate(completed: dict[str, dict[str, Any]]) -> dict[str, bool]:
            summary = evaluate_condition(
                completed, hidden_rows, answer_metadata, catalog,
                witness_config, comparator_config, config,
            )
            summary_holder["summary"] = summary
            return dict(summary["qualification_gates"])

        result = run_locked_census_once(
            output_dir=output_dir / name,
            attempt={"condition": name, "adaptive_order": list(config["conditions"])},
            fixture_rows=fixture_rows,
            evaluate_fixture=evaluator,
            evaluate_gates=aggregate,
            result_metadata={
                "schema_version": "154-local-question-order-condition-result",
                "experiment": config["experiment"],
                "condition": name,
                "development_request_fixture_count": len(public_rows),
                "claim_boundary": config["claimBoundary"],
            },
            pass_decision=f"{name}_passes_condition_gates",
            fail_decision=f"{name}_fails_condition_gates",
        )
        return result, summary_holder["summary"]

    direct_result, direct_summary = run_condition("direct", evaluate_direct)
    low_result = None
    low_summary = None
    conditions_run = ["direct"]
    if not direct_summary["qualified"]:
        low_result, low_summary = run_condition("bounded-low-reasoning", evaluate_bounded_low)
        conditions_run.append("bounded-low-reasoning")

    access_gates = config["accessGates"]
    access_checks = {
        "tokenizer_load_budget": access["tokenizer_load_count"] <= access_gates["maximumTokenizerLoadCount"],
        "model_load_budget": access["model_load_count"] <= access_gates["maximumModelLoadCount"],
        "direct_generation_budget": access["direct_generation_count"] <= access_gates["maximumDirectGenerationCount"],
        "bounded_reasoning_generation_budget": access["bounded_reasoning_phase_generation_count"] <= access_gates["maximumBoundedReasoningPhaseGenerationCount"],
        "bounded_final_generation_budget": access["bounded_final_phase_generation_count"] <= access_gates["maximumBoundedFinalPhaseGenerationCount"],
        "total_generation_budget": (
            access["direct_generation_count"]
            + access["bounded_reasoning_phase_generation_count"]
            + access["bounded_final_phase_generation_count"]
        ) <= access_gates["maximumTotalGenerationCount"],
        "per_condition_fixture_generation_budget": access["maximum_generation_calls_per_condition_fixture"] <= access_gates["maximumGenerationCallsPerConditionFixture"],
        "zero_closed_answer_generation": access["closed_answer_model_generation_count"] <= access_gates["maximumClosedAnswerModelGenerationCount"],
        "zero_evaluation_generation": access["evaluation_fixture_model_generation_count"] <= access_gates["maximumEvaluationFixtureModelGenerationCount"],
        "zero_retries": access["retry_count"] <= access_gates["maximumRetryCount"],
        "zero_manual_raw_inspection": access["manual_raw_response_inspection_count"] <= access_gates["maximumManualRawResponseInspectionCount"],
        "zero_persisted_raw": access["persisted_raw_response_count"] <= access_gates["maximumPersistedRawResponseCount"],
        "zero_API": access["API_call_count"] <= access_gates["maximumAPICallCount"],
        "zero_training": access["training_run_count"] <= access_gates["maximumTrainingRunCount"],
        "zero_services": access["real_service_call_count"] <= access_gates["maximumRealServiceCallCount"],
        "zero_side_effects": access["external_side_effect_count"] <= access_gates["maximumExternalSideEffectCount"],
        "zero_execution": access["actual_execution_count"] <= access_gates["maximumActualExecutionCount"],
    }
    access_pass = all(access_checks.values())
    if direct_summary["qualified"] and access_pass:
        selected = "direct"
        decision = config["decisionRule"]["ifDirectQualifies"]
    elif low_summary is not None and low_summary["qualified"] and access_pass:
        selected = "bounded-low-reasoning"
        decision = config["decisionRule"]["ifDirectFailsAndBoundedLowReasoningQualifies"]
    else:
        selected = None
        decision = config["decisionRule"]["otherwise"]
    persist_progress()
    result = {
        "schema_version": "154-adaptive-local-question-order-result",
        "experiment": config["experiment"],
        "completed_condition": True,
        "conditions_run": conditions_run,
        "bounded_low_reasoning_triggered": low_summary is not None,
        "selected_condition": selected,
        "decision": decision,
        "direct_summary": direct_summary,
        "bounded_low_reasoning_summary": low_summary,
        "access_gates": access_checks,
        "claim_boundary": config["claimBoundary"],
    }
    write_json(output_dir / "result.json", result)
    write_json(output_dir / "access.json", access)
    print(json.dumps({"result": result, "access": access}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
