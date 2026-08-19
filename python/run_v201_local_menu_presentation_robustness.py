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
from v154_adaptive_local_question_order import prepare_bounded_final_prompt_tokens
from v195_bounded_local_language_menu_ranker import parse_response, render_prompt
from v201_local_menu_presentation_robustness import evaluate_access_gates, evaluate_model
from v22r2_grounding import PROJECT_ROOT


def main() -> None:
    lock_path = PROJECT_ROOT / "configs/v201-local-menu-presentation-robustness-lock.json"
    output_dir = PROJECT_ROOT / "outputs/v201-local-menu-presentation-robustness/model-realization"
    if output_dir.exists(): raise RuntimeError("V201 exact realization may run only once")
    lock = json.loads(lock_path.read_text())
    if not valid_lock(lock): raise RuntimeError("V201 lock mismatch")
    for key in [key for key in lock if not key.endswith("_sha256") and f"{key}_sha256" in lock]:
        if file_sha256(PROJECT_ROOT / lock[key]) != lock[f"{key}_sha256"]:
            raise RuntimeError(f"V201 dependency drifted: {key}")
    auth = lock["authorization"]
    if not (auth["run_exact_single_local_development_robustness_realization"] and not auth["generate_on_missing_records_retry_or_select_outputs"] and not auth["persist_or_manually_inspect_raw_model_outputs"] and not auth["read_protected_language_or_run_API_training_registration_authority_action_or_execution"]):
        raise RuntimeError("V201 authorization invalid")
    config = lock["config_payload"]
    language = json.loads((PROJECT_ROOT / lock["development_language"]).read_text())
    hidden_targets = json.loads((PROJECT_ROOT / lock["hidden_targets"]).read_text())
    visible_variants = json.loads((PROJECT_ROOT / lock["visible_menu_variants"]).read_text())
    hidden_maps = json.loads((PROJECT_ROOT / lock["hidden_variant_maps"]).read_text())
    canonical_map = json.loads((PROJECT_ROOT / lock["canonical_hidden_option_map"]).read_text())
    canonical_census = json.loads((PROJECT_ROOT / lock["canonical_model_census"]).read_text())
    char_summary = json.loads((PROJECT_ROOT / lock["transformed_CHAR_LAST_summary"]).read_text())
    prior = json.loads((PROJECT_ROOT / lock["primary_prior"]).read_text())
    fixed_costs = json.loads((PROJECT_ROOT / lock["fixed_hierarchy_target_costs"]).read_text())
    manifest = json.loads((PROJECT_ROOT / lock["model_manifest"]).read_text())
    observed = [row for row in language["records"] if row["observation_available"]]
    missing = [row for row in language["records"] if not row["observation_available"]]
    visible_by_id = {row["record_id"]: row for row in visible_variants["records"]}
    fixture_rows = []
    for record in observed:
        for variant_id in config["trustedEvaluation"]["variantIds"]:
            menu = next(row for row in visible_by_id[record["record_id"]]["variants"] if row["variant_id"] == variant_id)
            fixture_rows.append({"name": f"{record['record_id']}@@{variant_id}", "record": record, "variant_id": variant_id, "menu": {"options": menu["options"]}})
    if len(fixture_rows) != config["population"]["requiredObservedRecordVariantGenerationCount"] or len(missing) * 2 != config["population"]["requiredMissingRecordVariantNoGenerationCount"]:
        raise RuntimeError("V201 population count mismatch")
    snapshot = Path(manifest["snapshot_path"])
    access: dict[str, Any] = {
        "development_language_read_count": len(language["records"]), "tokenizer_load_count": 0, "model_load_count": 0,
        "reasoning_phase_generation_count": 0, "final_phase_generation_count": 0, "total_generation_count": 0,
        "maximum_generation_calls_per_observed_record_variant": 0, "missing_record_variant_generation_count": 0,
        "retry_count": 0, "manual_raw_response_inspection_count": 0, "persisted_raw_response_count": 0,
        "protected_language_read_count": 0, "API_call_count": 0, "training_run_count": 0,
        "ontology_registration_count": 0, "trusted_state_mutation_count": 0, "real_service_call_count": 0,
        "external_side_effect_count": 0, "actual_execution_count": 0,
    }
    state: dict[str, Any] = {"model": None, "tokenizer": None}
    sampler = make_sampler(temp=config["model"]["temperature"]); mx.reset_peak_memory(); started = time.perf_counter()
    def persist_progress() -> None:
        access["total_generation_count"] = access["reasoning_phase_generation_count"] + access["final_phase_generation_count"]
        access["elapsed_seconds"] = time.perf_counter() - started; access["peak_active_memory_bytes"] = int(mx.get_peak_memory())
        write_json(output_dir / "access-progress.json", access)
    def ensure_loaded() -> None:
        if state["model"] is not None: return
        load_started = time.perf_counter(); model, tokenizer = load(str(snapshot)); model.eval()
        state["model"], state["tokenizer"] = model, tokenizer; access["model_load_count"] += 1; access["tokenizer_load_count"] += 1
        access["model_load_seconds"] = time.perf_counter() - load_started; persist_progress()
    def evaluate_fixture(row: dict[str, Any]) -> dict[str, Any]:
        ensure_loaded(); record = row["record"]; menu = row["menu"]
        payload = render_prompt(menu, record, config)
        messages = [{"role": "system", "content": config["prompt"]["system"]}, {"role": "user", "content": payload}]
        prompt = state["tokenizer"].apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=config["model"]["enableThinking"], reasoning_effort=config["model"]["reasoningEffort"])
        prompt_tokens = list(state["tokenizer"].encode(prompt, add_special_tokens=False))
        if len(prompt_tokens) > config["prompt"]["maximumPromptTokens"]: raise RuntimeError("V201 prompt exceeds frozen budget")
        access["reasoning_phase_generation_count"] += 1; access["maximum_generation_calls_per_observed_record_variant"] = 2; persist_progress()
        generation_started = time.perf_counter()
        responses = list(stream_generate(state["model"], state["tokenizer"], prompt=prompt_tokens, max_tokens=config["model"]["reasoningPhaseMaximumTokens"], sampler=sampler))
        reasoning_tokens = [response.token for response in responses]; reasoning_text = state["tokenizer"].decode(reasoning_tokens)
        final_prompt_tokens, natural_close, retained_count = prepare_bounded_final_prompt_tokens(prompt_tokens, reasoning_tokens, state["tokenizer"])
        access["final_phase_generation_count"] += 1; persist_progress()
        raw = generate(state["model"], state["tokenizer"], prompt=final_prompt_tokens, max_tokens=config["model"]["finalPhaseMaximumTokens"], sampler=sampler, verbose=False)
        final_tokens = list(state["tokenizer"].encode(raw, add_special_tokens=False)); valid_ids = {option["option_id"] for option in menu["options"]}
        parsed = parse_response(raw, valid_ids); persist_progress()
        return {
            "name": row["name"], "record_id": record["record_id"], "variant_id": row["variant_id"], **parsed,
            "reasoning_response_sha256": hashlib.sha256(reasoning_text.encode()).hexdigest(), "raw_response_sha256": hashlib.sha256(raw.encode()).hexdigest(),
            "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(), "bounded_final_prompt_sha256": hashlib.sha256(bytes(str(final_prompt_tokens), "utf-8")).hexdigest(),
            "prompt_token_count": len(prompt_tokens), "reasoning_phase_generated_token_count": len(reasoning_tokens),
            "reasoning_phase_retained_token_count": retained_count, "reasoning_naturally_closed_within_budget": natural_close,
            "reasoning_phase_maximum_tokens_hit": len(reasoning_tokens) >= config["model"]["reasoningPhaseMaximumTokens"],
            "final_phase_generated_token_count": len(final_tokens), "final_phase_maximum_tokens_hit": len(final_tokens) >= config["model"]["finalPhaseMaximumTokens"],
            "generation_seconds": time.perf_counter() - generation_started, "raw_response_persisted": False,
        }
    holder: dict[str, Any] = {}
    def aggregate(completed: dict[str, dict[str, Any]]) -> dict[str, bool]:
        persist_progress(); evaluation = evaluate_model(completed, language, hidden_targets, hidden_maps, canonical_map, canonical_census, char_summary, prior, fixed_costs, access, config); holder.update(evaluation)
        return {f"{row['variant_id']}::{gate}": passed for row in evaluation["summary"]["variants"] for gate, passed in row["qualification_gates"].items()}
    census = run_locked_census_once(
        output_dir=output_dir / "census",
        attempt={"condition": "unchanged-confirmed-policy-exact-menu-presentation-shift", "model_repository": config["model"]["repository"], "model_revision": config["model"]["revision"], "variant_ids": config["trustedEvaluation"]["variantIds"], "reasoning_phase_maximum_tokens": 48, "final_phase_maximum_tokens": 64},
        fixture_rows=fixture_rows, evaluate_fixture=evaluate_fixture, evaluate_gates=aggregate,
        result_metadata={"schema_version": "201-local-menu-presentation-robustness-census-result", "experiment": config["experiment"], "observed_record_variant_count": len(fixture_rows), "missing_record_variant_count": len(missing) * 2, "claim_boundary": config["claimBoundary"]},
        pass_decision="V201_per_variant_qualification_gates_pass", fail_decision="V201_one_or_more_per_variant_qualification_gates_fail",
    )
    access_checks = evaluate_access_gates(access, config); access_pass = all(access_checks.values()); qualified = bool(holder["summary"]["qualified"] and access_pass)
    decision = config["decisionRule"]["ifEveryPerVariantQualificationAndAccessGatePasses" if qualified else "otherwise"]
    persist_progress(); write_json(output_dir / "evaluation-summary.json", holder["summary"]); write_json(output_dir / "scored-records.json", holder["scored_records"]); write_json(output_dir / "access.json", access)
    result = {"schema_version": "201-local-menu-presentation-robustness-result", "experiment": config["experiment"], "completed": True, "qualification_gates_passed": holder["summary"]["qualified"], "access_gates_passed": access_pass, "qualified": qualified, "decision": decision, "access_gates": access_checks, "summary": holder["summary"], "census_passed": census["passed"], "claim_boundary": config["claimBoundary"]}
    write_json(output_dir / "result.json", result)
    print(json.dumps({"qualified": qualified, "decision": decision, "variants": result["summary"]["variants"], "elapsed_seconds": access["elapsed_seconds"]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
