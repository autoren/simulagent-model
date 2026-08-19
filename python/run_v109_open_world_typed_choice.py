#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time
from typing import Any

import mlx.core as mx
from mlx_lm import generate, load
from mlx_lm.sample_utils import make_sampler

from locked_census_harness import run_locked_census_once, write_json
from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT
from v106_open_world_benchmark import split_development_records
from v109_open_world_typed_choice import (
    evaluate_v109_gates, render_choice_prompt, validate_and_expand_choice,
)


def payload_hash(value: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def read_jsonl_bytes(value: bytes) -> list[dict[str, Any]]:
    return [json.loads(line) for line in value.decode().splitlines() if line]


def prepare_fixture_rows(lock: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], set[str]]:
    development_path = PROJECT_ROOT / lock["development_language"]
    development_bytes = development_path.read_bytes()
    if hashlib.sha256(development_bytes).hexdigest() != lock["development_language_sha256"]:
        raise RuntimeError("V109 development language identity mismatch")
    split = split_development_records(read_jsonl_bytes(development_bytes), lock["baseline_config_payload"])
    holdback = split["calibration"]
    controlled = json.loads((PROJECT_ROOT / lock["controlled_identifiers"]).read_text())
    control_rows = controlled["role_records"]["development"]
    controlled_ids = {row["controlled_record_id"] for row in control_rows}
    fixtures = [
        {"name": row["record_id"], "kind": "observed_model_blind_holdback", "record": row}
        for row in holdback
    ] + [
        {"name": row["controlled_record_id"], "kind": "controlled_missing_observation", "record": None}
        for row in sorted(control_rows, key=lambda value: value["controlled_record_id"])
    ]
    return fixtures, holdback, controlled_ids


def decision_for(interface_pass: bool, semantic_pass: bool, access_pass: bool) -> str:
    if not access_pass:
        return "condition_invalid_due_to_access_gate_failure"
    if interface_pass and semantic_pass:
        return "typed_choice_interface_and_semantics_pass_preregister_sequential_clarification"
    if interface_pass:
        return "serialization_repaired_residual_semantic_risk_preregister_sequential_clarification"
    return "typed_choice_interface_nonviable_close_model_serialization_branch"


def main() -> None:
    lock_path = PROJECT_ROOT / "configs/v109-open-world-typed-choice-implementation-lock.json"
    lock = json.loads(lock_path.read_text())
    if payload_hash({key: value for key, value in lock.items() if key != "lock_payload_sha256"}) != lock["lock_payload_sha256"]:
        raise RuntimeError("V109 implementation lock mismatch")
    dependency_keys = (
        "config", "parent_forensics_outcome", "forensics_lock", "V107_outcome",
        "V107_implementation_lock", "V107_result", "baseline_outcome", "baseline_lock",
        "development_membership", "development_language", "visible_catalog",
        "controlled_identifiers", "model_manifest", "choice_catalog", "plan", "protocol",
        "tests", "runner", "verifier", "auditor", "census_harness", "implementation_audit",
    )
    for key in dependency_keys:
        if file_sha256(PROJECT_ROOT / lock[key]) != lock[f"{key}_sha256"]:
            raise RuntimeError(f"V109 dependency drifted: {key}")
    config = lock["config_payload"]
    baseline_config = lock["baseline_config_payload"]
    choice_catalog = json.loads((PROJECT_ROOT / lock["choice_catalog"]).read_text())
    manifest = json.loads((PROJECT_ROOT / lock["model_manifest"]).read_text())
    snapshot = Path(manifest["snapshot_path"])
    if not snapshot.is_dir() or snapshot.name != config["condition"]["revision"]:
        raise RuntimeError("V109 pinned model snapshot unavailable")
    fixture_rows, holdback_records, controlled_ids = prepare_fixture_rows(lock)
    if len(fixture_rows) != config["corpus"]["totalGenerationCount"]:
        raise RuntimeError("V109 fixture count mismatch")
    output_dir = PROJECT_ROOT / "outputs/v109-open-world-typed-choice/holdback-evaluation"
    access: dict[str, Any] = {
        "condition_id": config["condition"]["id"],
        "development_language_read_count": 1, "protected_test_language_read_count": 0,
        "manual_utterance_inspection_count": 0, "model_load_count": 0,
        "model_generation_count": 0, "LLM_API_call_count": 0,
        "adapter_training_run_count": 0, "real_service_call_count": 0,
        "external_side_effect_count": 0,
    }
    state: dict[str, Any] = {"model": None, "tokenizer": None}
    mx.reset_peak_memory()
    condition_start = time.perf_counter()

    def evaluate_fixture(fixture: dict[str, Any]) -> dict[str, Any]:
        if state["model"] is None:
            load_start = time.perf_counter()
            model, tokenizer = load(str(snapshot))
            model.eval()
            state["model"] = model
            state["tokenizer"] = tokenizer
            access["model_load_count"] += 1
            access["model_load_seconds"] = time.perf_counter() - load_start
        observed = fixture["kind"] == "observed_model_blind_holdback"
        utterance = fixture["record"]["utterance"] if observed else None
        user_payload = render_choice_prompt(choice_catalog, utterance, observed, config)
        tokenizer = state["tokenizer"]
        prompt = tokenizer.apply_chat_template(
            [
                {"role": "system", "content": config["prompt"]["system"]},
                {"role": "user", "content": user_payload},
            ],
            tokenize=False, add_generation_prompt=True, enable_thinking=False,
        )
        prompt_tokens = tokenizer.encode(prompt)
        if len(prompt_tokens) > config["prompt"]["maximumPromptTokens"]:
            raise RuntimeError("V109 prompt exceeds frozen token budget")
        generation_start = time.perf_counter()
        raw_response = generate(
            state["model"], tokenizer, prompt=prompt,
            max_tokens=config["decoding"]["maximumNewTokens"],
            sampler=make_sampler(temp=config["decoding"]["temperature"]), verbose=False,
        )
        generation_seconds = time.perf_counter() - generation_start
        access["model_generation_count"] += 1
        access["elapsed_seconds"] = time.perf_counter() - condition_start
        access["peak_active_memory_bytes"] = int(mx.get_peak_memory())
        write_json(output_dir / "access-progress.json", access)
        parsed, valid, reason = validate_and_expand_choice(raw_response, choice_catalog, config)
        return {
            "name": fixture["name"], "kind": fixture["kind"],
            "raw_response": raw_response,
            "raw_response_sha256": hashlib.sha256(raw_response.encode()).hexdigest(),
            "parsed_response": parsed, "response_valid": valid,
            "validation_reason": reason, "prompt_token_count": len(prompt_tokens),
            "generated_token_count": len(tokenizer.encode(raw_response)),
            "permanently_non_authoritative": True,
            "safe_hypothesis_universe_pruned": False, "executable": False,
        }

    def gates(fixtures: dict[str, dict[str, Any]]) -> dict[str, bool]:
        metrics, interface_checks, semantic_checks, access_checks = evaluate_v109_gates(
            fixtures, holdback_records, controlled_ids, choice_catalog, access,
            baseline_config, config,
        )
        gates.last_metrics = metrics
        gates.interface_checks = interface_checks
        gates.semantic_checks = semantic_checks
        gates.access_checks = access_checks
        return {
            **{f"interface::{key}": value for key, value in interface_checks.items()},
            **{f"semantic::{key}": value for key, value in semantic_checks.items()},
            **{f"access::{key}": value for key, value in access_checks.items()},
        }

    gates.last_metrics = None
    gates.interface_checks = {}
    gates.semantic_checks = {}
    gates.access_checks = {}
    result = run_locked_census_once(
        output_dir=output_dir, attempt=access, fixture_rows=fixture_rows,
        evaluate_fixture=evaluate_fixture, evaluate_gates=gates,
        result_metadata={
            "schema_version": "109-open-world-typed-choice-development-result",
            "experiment": config["experiment"], "condition": config["condition"],
            "model_manifest_sha256": manifest["manifest_sha256"],
            "claim_boundary": "model-blind development holdback; single typed choice expanded deterministically; non-authoritative shadow only; no protected test, API, training, action, service call, or side effect",
        },
        pass_decision="typed_choice_all_gates_pass",
        fail_decision="typed_choice_one_or_more_gates_fail",
    )
    access["elapsed_seconds"] = time.perf_counter() - condition_start
    access["peak_active_memory_bytes"] = int(mx.get_peak_memory())
    interface_pass = all(gates.interface_checks.values())
    semantic_pass = all(gates.semantic_checks.values())
    access_pass = all(gates.access_checks.values())
    result.update({
        "metrics": gates.last_metrics, "interface_gates": gates.interface_checks,
        "semantic_gates": gates.semantic_checks, "access_gates": gates.access_checks,
        "interface_pass": interface_pass, "semantic_pass": semantic_pass,
        "access_pass": access_pass,
        "decision": decision_for(interface_pass, semantic_pass, access_pass),
        "regret_above_ask_always": gates.last_metrics["mean_regret"] - 1.125,
        "completed_condition": bool(
            len(result["fixtures"]) == config["accessGates"]["requiredFixtureCount"]
            and access["model_load_count"] == 1
            and access["model_generation_count"] == config["corpus"]["totalGenerationCount"]
        ),
        "final_access": access,
    })
    write_json(output_dir / "result.json", result)
    write_json(output_dir / "access.json", access)
    print(json.dumps({
        "completed_condition": result["completed_condition"],
        "interface_pass": interface_pass, "semantic_pass": semantic_pass,
        "access_pass": access_pass, "decision": result["decision"],
        "metrics": result["metrics"], "interface_gates": result["interface_gates"],
        "semantic_gates": result["semantic_gates"], "access_gates": result["access_gates"],
        "access": access,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
