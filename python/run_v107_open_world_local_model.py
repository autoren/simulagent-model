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
from v105_open_world_interface import render_prompt, validate_response
from v106_open_world_benchmark import split_development_records
from v107_open_world_local_model import (
    aggregate_model_fixtures, evaluate_model_gates, quality_gate_pass,
)


def payload_hash(value: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def read_jsonl_bytes(value: bytes) -> list[dict[str, Any]]:
    return [json.loads(line) for line in value.decode().splitlines() if line]


def prepare_fixture_rows(lock: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], set[str]]:
    development_bytes = (PROJECT_ROOT / lock["development_language"]).read_bytes()
    if hashlib.sha256(development_bytes).hexdigest() != lock["development_language_sha256"]:
        raise RuntimeError("V107 development language identity mismatch")
    development_records = read_jsonl_bytes(development_bytes)
    split = split_development_records(development_records, lock["baseline_config_payload"])
    evaluation = split["evaluation"]
    controlled = json.loads((PROJECT_ROOT / lock["controlled_identifiers"]).read_text())
    control_rows = controlled["role_records"]["development"]
    control_ids = {row["controlled_record_id"] for row in control_rows}
    fixtures = [
        {"name": row["record_id"], "kind": "observed_evaluation", "record": row}
        for row in evaluation
    ] + [
        {"name": row["controlled_record_id"], "kind": "controlled_missing_observation", "record": None}
        for row in sorted(control_rows, key=lambda value: value["controlled_record_id"])
    ]
    return fixtures, evaluation, control_ids


def main() -> None:
    lock_path = PROJECT_ROOT / "configs/v107-open-world-local-model-implementation-lock.json"
    lock = json.loads(lock_path.read_text())
    if payload_hash({key: value for key, value in lock.items() if key != "lock_payload_sha256"}) != lock["lock_payload_sha256"]:
        raise RuntimeError("V107 implementation lock mismatch")
    dependency_keys = (
        "config", "parent_baseline_outcome", "baseline_lock", "interface_outcome",
        "interface_lock", "visible_catalog", "controlled_identifiers", "model_manifest",
        "plan", "protocol", "tests", "runner", "verifier", "auditor", "census_harness",
        "implementation_audit",
    )
    for key in dependency_keys:
        if file_sha256(PROJECT_ROOT / lock[key]) != lock[f"{key}_sha256"]:
            raise RuntimeError(f"V107 dependency drifted: {key}")
    config = lock["config_payload"]
    baseline_config = lock["baseline_config_payload"]
    interface_config = lock["interface_config_payload"]
    manifest = json.loads((PROJECT_ROOT / lock["model_manifest"]).read_text())
    snapshot = Path(manifest["snapshot_path"])
    if not snapshot.is_dir() or snapshot.name != config["condition"]["revision"]:
        raise RuntimeError("V107 pinned snapshot is unavailable")
    catalog = json.loads((PROJECT_ROOT / lock["visible_catalog"]).read_text())
    fixture_rows, evaluation_records, controlled_ids = prepare_fixture_rows(lock)
    if len(fixture_rows) != config["corpus"]["totalGenerationCount"]:
        raise RuntimeError("V107 fixture count mismatch")
    parent = json.loads((PROJECT_ROOT / lock["parent_baseline_outcome"]).read_text())
    best_baseline_regret = parent["outcome"]["development_summary"]["best_nonoracle_baseline"]["mean_regret"]
    output_dir = PROJECT_ROOT / "outputs/v107-open-world-local-model/development-evaluation"
    access: dict[str, Any] = {
        "condition_id": config["condition"]["id"],
        "development_language_read_count": 1,
        "protected_test_language_read_count": 0,
        "manual_utterance_inspection_count": 0,
        "model_load_count": 0, "model_generation_count": 0,
        "LLM_API_call_count": 0, "adapter_training_run_count": 0,
        "real_service_call_count": 0, "external_side_effect_count": 0,
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
        observed = fixture["kind"] == "observed_evaluation"
        utterance = fixture["record"]["utterance"] if observed else None
        user_payload = render_prompt(catalog, utterance, observed, interface_config)
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
            raise RuntimeError("V107 prompt exceeds locked token budget")
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
        parsed, valid, reason = validate_response(raw_response, catalog, interface_config)
        return {
            "name": fixture["name"], "kind": fixture["kind"],
            "raw_response": raw_response,
            "raw_response_sha256": hashlib.sha256(raw_response.encode()).hexdigest(),
            "parsed_response": parsed, "response_valid": valid,
            "validation_reason": reason, "prompt_token_count": len(prompt_tokens),
            "generated_token_count": len(tokenizer.encode(raw_response)),
            "generation_seconds": generation_seconds,
            "permanently_non_authoritative": True,
            "safe_hypothesis_universe_pruned": False,
            "executable": False,
        }

    def gates(fixtures: dict[str, dict[str, Any]]) -> dict[str, bool]:
        metrics = aggregate_model_fixtures(
            fixtures, evaluation_records, controlled_ids, baseline_config,
        )
        gates.last_metrics = metrics
        checks = evaluate_model_gates(metrics, best_baseline_regret, access, config)
        gates.last_quality_pass = quality_gate_pass(checks)
        return checks

    gates.last_metrics = None
    gates.last_quality_pass = False
    result = run_locked_census_once(
        output_dir=output_dir, attempt=access, fixture_rows=fixture_rows,
        evaluate_fixture=evaluate_fixture, evaluate_gates=gates,
        result_metadata={
            "schema_version": "107-open-world-local-model-development-result",
            "experiment": config["experiment"], "condition": config["condition"],
            "model_manifest_sha256": manifest["manifest_sha256"],
            "best_nonoracle_baseline_regret": best_baseline_regret,
            "claim_boundary": "development-only offline shadow semantic proposal; complete safe hypotheses retained; no protected test, API, training, action, tool, service call, or side effect",
        },
        pass_decision="model_qualifies_for_separately_preregistered_protected_test_only",
        fail_decision="model_is_nonqualifying_and_protected_test_remains_sealed",
    )
    access["elapsed_seconds"] = time.perf_counter() - condition_start
    access["peak_active_memory_bytes"] = int(mx.get_peak_memory())
    result["metrics"] = gates.last_metrics
    result["quality_gate_pass"] = gates.last_quality_pass
    result["regret_above_best_nonoracle_baseline"] = (
        gates.last_metrics["mean_regret"] - best_baseline_regret
    )
    result["completed_condition"] = bool(
        len(result["fixtures"]) == config["accessGates"]["requiredFixtureCount"]
        and access["model_load_count"] == 1
        and access["model_generation_count"] == config["corpus"]["totalGenerationCount"]
    )
    result["final_access"] = access
    write_json(output_dir / "result.json", result)
    write_json(output_dir / "access.json", access)
    print(json.dumps({
        "completed_condition": result["completed_condition"],
        "quality_gate_pass": result["quality_gate_pass"], "decision": result["decision"],
        "metrics": result["metrics"], "regret_above_best_nonoracle_baseline": result["regret_above_best_nonoracle_baseline"],
        "gates": result["gates"], "access": access,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
