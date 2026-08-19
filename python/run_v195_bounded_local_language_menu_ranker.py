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

from audit_and_freeze_v165_factored_ontology_identifiability_population import valid_lock
from locked_census_harness import run_locked_census_once, write_json
from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT
from v154_adaptive_local_question_order import prepare_bounded_final_prompt_tokens
from v195_bounded_local_language_menu_ranker import (
    evaluate_access_gates,
    evaluate_model,
    parse_response,
    render_prompt,
)


def main() -> None:
    lock_path = PROJECT_ROOT / "configs/v195-bounded-local-language-menu-ranker-lock.json"
    output_dir = PROJECT_ROOT / "outputs/v195-bounded-local-language-menu-ranker/model-realization"
    if output_dir.exists():
        raise RuntimeError("V195 exact local development realization may run only once")
    lock = json.loads(lock_path.read_text())
    if not valid_lock(lock):
        raise RuntimeError("V195 lock mismatch")
    dependencies = [key for key in lock if not key.endswith("_sha256") and f"{key}_sha256" in lock]
    for key in dependencies:
        if file_sha256(PROJECT_ROOT / lock[key]) != lock[f"{key}_sha256"]:
            raise RuntimeError(f"V195 dependency drifted: {key}")
    auth = lock["authorization"]
    if not (
        auth["run_exact_single_bounded_local_development_realization"]
        and not auth["modify_population_prompt_model_decode_budgets_parser_costs_or_gates"]
        and not auth["generate_on_missing_fixtures_or_retry"]
        and not auth["persist_or_manually_inspect_raw_model_outputs"]
        and not auth["run_API_training_protected_access_registration_authority_action_or_execution"]
    ):
        raise RuntimeError("V195 authorization is invalid")

    config = lock["config_payload"]
    language = json.loads((PROJECT_ROOT / lock["development_language"]).read_text())
    hidden_targets = json.loads((PROJECT_ROOT / lock["hidden_targets"]).read_text())
    menu = json.loads((PROJECT_ROOT / lock["visible_menu"]).read_text())
    option_map = json.loads((PROJECT_ROOT / lock["hidden_option_map"]).read_text())
    prior = json.loads((PROJECT_ROOT / lock["primary_prior"]).read_text())
    fixed_costs = json.loads((PROJECT_ROOT / lock["fixed_hierarchy_target_costs"]).read_text())
    deterministic_results = json.loads((PROJECT_ROOT / lock["deterministic_ranker_results"]).read_text())
    manifest = json.loads((PROJECT_ROOT / lock["model_manifest"]).read_text())
    observed = [row for row in language["records"] if row["observation_available"]]
    missing = [row for row in language["records"] if not row["observation_available"]]
    if len(observed) != config["population"]["requiredObservedGenerationCount"] or len(missing) != config["population"]["requiredMissingNoGenerationCount"]:
        raise RuntimeError("V195 locked population count mismatch")
    snapshot = Path(manifest["snapshot_path"])
    valid_ids = {row["option_id"] for row in menu["options"]}
    fixture_rows = [{"name": row["record_id"], "record": row} for row in observed]

    access: dict[str, Any] = {
        "tokenizer_load_count": 0,
        "model_load_count": 0,
        "reasoning_phase_generation_count": 0,
        "final_phase_generation_count": 0,
        "total_generation_count": 0,
        "maximum_generation_calls_per_observed_fixture": 0,
        "missing_fixture_generation_count": 0,
        "retry_count": 0,
        "manual_raw_response_inspection_count": 0,
        "persisted_raw_response_count": 0,
        "protected_language_read_count": 0,
        "API_call_count": 0,
        "training_run_count": 0,
        "ontology_registration_count": 0,
        "trusted_state_mutation_count": 0,
        "real_service_call_count": 0,
        "external_side_effect_count": 0,
        "actual_execution_count": 0,
    }
    state: dict[str, Any] = {"model": None, "tokenizer": None}
    sampler = make_sampler(temp=config["model"]["temperature"])
    mx.reset_peak_memory()
    started = time.perf_counter()

    def persist_progress() -> None:
        access["total_generation_count"] = access["reasoning_phase_generation_count"] + access["final_phase_generation_count"]
        access["elapsed_seconds"] = time.perf_counter() - started
        access["peak_active_memory_bytes"] = int(mx.get_peak_memory())
        write_json(output_dir / "access-progress.json", access)

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
        persist_progress()

    def evaluate_fixture(row: dict[str, Any]) -> dict[str, Any]:
        ensure_loaded()
        record = row["record"]
        payload = render_prompt(menu, record, config)
        messages = [
            {"role": "system", "content": config["prompt"]["system"]},
            {"role": "user", "content": payload},
        ]
        prompt = state["tokenizer"].apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=config["model"]["enableThinking"],
            reasoning_effort=config["model"]["reasoningEffort"],
        )
        prompt_tokens = list(state["tokenizer"].encode(prompt, add_special_tokens=False))
        if len(prompt_tokens) > config["prompt"]["maximumPromptTokens"]:
            raise RuntimeError("V195 prompt exceeds frozen token budget")

        access["reasoning_phase_generation_count"] += 1
        access["maximum_generation_calls_per_observed_fixture"] = max(
            access["maximum_generation_calls_per_observed_fixture"], 2
        )
        persist_progress()
        generation_started = time.perf_counter()
        responses = list(stream_generate(
            state["model"], state["tokenizer"], prompt=prompt_tokens,
            max_tokens=config["model"]["reasoningPhaseMaximumTokens"], sampler=sampler,
        ))
        reasoning_tokens = [response.token for response in responses]
        reasoning_text = state["tokenizer"].decode(reasoning_tokens)
        final_prompt_tokens, natural_close, retained_count = prepare_bounded_final_prompt_tokens(
            prompt_tokens, reasoning_tokens, state["tokenizer"]
        )

        access["final_phase_generation_count"] += 1
        persist_progress()
        raw = generate(
            state["model"], state["tokenizer"], prompt=final_prompt_tokens,
            max_tokens=config["model"]["finalPhaseMaximumTokens"], sampler=sampler, verbose=False,
        )
        elapsed = time.perf_counter() - generation_started
        final_tokens = list(state["tokenizer"].encode(raw, add_special_tokens=False))
        parsed = parse_response(raw, valid_ids)
        persist_progress()
        return {
            "name": row["name"],
            "record_id": record["record_id"],
            **parsed,
            "reasoning_response_sha256": hashlib.sha256(reasoning_text.encode()).hexdigest(),
            "raw_response_sha256": hashlib.sha256(raw.encode()).hexdigest(),
            "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
            "bounded_final_prompt_sha256": hashlib.sha256(bytes(str(final_prompt_tokens), "utf-8")).hexdigest(),
            "prompt_token_count": len(prompt_tokens),
            "reasoning_phase_generated_token_count": len(reasoning_tokens),
            "reasoning_phase_retained_token_count": retained_count,
            "reasoning_naturally_closed_within_budget": natural_close,
            "reasoning_phase_maximum_tokens_hit": len(reasoning_tokens) >= config["model"]["reasoningPhaseMaximumTokens"],
            "final_phase_generated_token_count": len(final_tokens),
            "final_phase_maximum_tokens_hit": len(final_tokens) >= config["model"]["finalPhaseMaximumTokens"],
            "generation_seconds": elapsed,
            "raw_response_persisted": False,
        }

    evaluation_holder: dict[str, Any] = {}

    def aggregate(completed: dict[str, dict[str, Any]]) -> dict[str, bool]:
        persist_progress()
        evaluation = evaluate_model(
            completed, language, hidden_targets, option_map, prior, fixed_costs,
            deterministic_results, access, config,
        )
        evaluation_holder.update(evaluation)
        return dict(evaluation["summary"]["qualification_gates"])

    census = run_locked_census_once(
        output_dir=output_dir / "census",
        attempt={
            "condition": "bounded-low-reasoning",
            "model_repository": config["model"]["repository"],
            "model_revision": config["model"]["revision"],
            "reasoning_phase_maximum_tokens": config["model"]["reasoningPhaseMaximumTokens"],
            "final_phase_maximum_tokens": config["model"]["finalPhaseMaximumTokens"],
        },
        fixture_rows=fixture_rows,
        evaluate_fixture=evaluate_fixture,
        evaluate_gates=aggregate,
        result_metadata={
            "schema_version": "195-bounded-local-language-menu-ranker-census-result",
            "experiment": config["experiment"],
            "observed_fixture_count": len(observed),
            "missing_fixture_count": len(missing),
            "claim_boundary": config["claimBoundary"],
        },
        pass_decision="V195_qualification_gates_pass",
        fail_decision="V195_qualification_gates_fail",
    )
    access_checks = evaluate_access_gates(access, config)
    access_pass = all(access_checks.values())
    qualified = bool(evaluation_holder["summary"]["qualified"] and access_pass)
    decision = (
        config["decisionRule"]["ifEveryQualificationAndAccessGatePasses"]
        if qualified else config["decisionRule"]["otherwise"]
    )
    persist_progress()
    write_json(output_dir / "evaluation-summary.json", evaluation_holder["summary"])
    write_json(output_dir / "scored-records.json", evaluation_holder["scored_records"])
    write_json(output_dir / "access.json", access)
    result = {
        "schema_version": "195-bounded-local-language-menu-ranker-result",
        "experiment": config["experiment"],
        "completed": True,
        "qualification_gates_passed": evaluation_holder["summary"]["qualified"],
        "access_gates_passed": access_pass,
        "qualified": qualified,
        "decision": decision,
        "qualification_gates": evaluation_holder["summary"]["qualification_gates"],
        "access_gates": access_checks,
        "summary": evaluation_holder["summary"],
        "census_passed": census["passed"],
        "claim_boundary": config["claimBoundary"],
    }
    write_json(output_dir / "result.json", result)
    print(json.dumps({
        "qualified": qualified,
        "decision": decision,
        "primary_top3_recall": result["summary"]["primary_top3_recall"],
        "primary_top3_mean_cost": result["summary"]["primary_top3_mean_cost"],
        "incremental_primary_improvement": result["summary"]["incremental_primary_improvement_over_deterministic_champion"],
        "structural_validity": result["summary"]["observed_structural_validity_rate"],
        "final_truncation_rate": result["summary"]["final_phase_token_limit_hit_rate"],
        "elapsed_seconds": access["elapsed_seconds"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
