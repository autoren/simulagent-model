#!/usr/bin/env python3
"""Run the one locked V91 local rank-only condition exactly once."""
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
from v91_rank_only_protocol import (
    aggregate_model_rows,
    evaluate_gates,
    format_user_prompt,
    score_response,
)


def payload_hash(value: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def main() -> None:
    lock_path = PROJECT_ROOT / "configs/v91-rank-only-implementation-lock.json"
    lock = json.loads(lock_path.read_text())
    lock_payload = {
        key: value for key, value in lock.items() if key != "lock_payload_sha256"
    }
    if payload_hash(lock_payload) != lock["lock_payload_sha256"]:
        raise RuntimeError("V91 implementation lock payload mismatch")
    for key in (
        "design_lock",
        "corpus_seal",
        "corpus",
        "protocol",
        "tests",
        "runner",
        "census_harness",
        "model_manifest",
        "planner_invariance_result",
        "implementation_auditor",
        "implementation_audit",
    ):
        if file_sha256(PROJECT_ROOT / lock[key]) != lock[f"{key}_sha256"]:
            raise RuntimeError(f"V91 implementation dependency drifted: {key}")
    if not lock["authorization"]["run_one_local_rank_only_census_once"]:
        raise RuntimeError("V91 local rank-only census is not authorized")

    config = lock["config_payload"]
    manifest = json.loads((PROJECT_ROOT / lock["model_manifest"]).read_text())
    model = config["modelCondition"]
    if (
        manifest["repository"] != model["repository"]
        or manifest["revision"] != model["revision"]
        or manifest["manifest_sha256"] != model["reuseManifestPayloadSha256"]
    ):
        raise RuntimeError("V91 reused model manifest identity drifted")
    snapshot = Path(manifest["snapshot_path"])
    if not snapshot.is_dir() or snapshot.name != model["revision"]:
        raise RuntimeError("V91 pinned local snapshot is unavailable")

    records = read_jsonl(PROJECT_ROOT / lock["corpus"])
    output_dir = PROJECT_ROOT / "outputs/v91-rank-only/evaluation"
    planner = json.loads((PROJECT_ROOT / lock["planner_invariance_result"]).read_text())
    access = {
        "source_language_record_access_count": len(records),
        "manual_utterance_inspection_count": 0,
        "new_model_weight_download_count": 0,
        "model_load_count": 0,
        "model_generation_count": 0,
        "LLM_API_call_count": 0,
        "adapter_training_run_count": 0,
        "real_service_call_count": 0,
        "external_side_effect_count": 0,
        "pruned_hypothesis_count": 0,
        "early_stopping_count": 0,
        "belief_update_from_model_count": 0,
        "action_selection_from_model_count": 0,
    }
    state: dict[str, Any] = {"model": None, "tokenizer": None}
    mx.reset_peak_memory()
    start = time.perf_counter()

    def evaluate_record(record: dict[str, Any]) -> dict[str, Any]:
        if state["model"] is None:
            load_start = time.perf_counter()
            loaded_model, tokenizer = load(str(snapshot))
            loaded_model.eval()
            state["model"] = loaded_model
            state["tokenizer"] = tokenizer
            access["model_load_count"] += 1
            access["model_load_seconds"] = time.perf_counter() - load_start
        tokenizer = state["tokenizer"]
        user_prompt = format_user_prompt(record, config)
        prompt = tokenizer.apply_chat_template(
            [
                {"role": "system", "content": config["systemPrompt"]},
                {"role": "user", "content": user_prompt},
            ],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        prompt_tokens = tokenizer.encode(prompt)
        if len(prompt_tokens) > config["decoding"]["maximumPromptTokens"]:
            raise RuntimeError("V91 prompt exceeds the locked token budget")
        generation_start = time.perf_counter()
        response = generate(
            state["model"],
            tokenizer,
            prompt=prompt,
            max_tokens=config["decoding"]["maximumNewTokens"],
            sampler=make_sampler(temp=config["decoding"]["temperature"]),
            verbose=False,
        )
        generation_seconds = time.perf_counter() - generation_start
        access["model_generation_count"] += 1
        access["elapsed_seconds"] = time.perf_counter() - start
        access["peak_active_memory_bytes"] = int(mx.get_peak_memory())
        write_json(output_dir / "access-progress.json", access)
        row = score_response(record, response)
        row.update(
            {
                "name": record["id"],
                "condition_id": model["id"],
                "prompt_token_count": len(prompt_tokens),
                "generated_token_count": len(tokenizer.encode(response)),
                "generation_seconds": generation_seconds,
            }
        )
        return row

    def gates(fixtures: dict[str, dict[str, Any]]) -> dict[str, bool]:
        metrics = aggregate_model_rows(list(fixtures.values()), records)
        gates.last_metrics = metrics
        checks = evaluate_gates(metrics, planner, config, access)
        gates.last_quality_pass = all(checks.values())
        return checks

    gates.last_metrics = None
    gates.last_quality_pass = False
    result = run_locked_census_once(
        output_dir=output_dir,
        attempt=access,
        fixture_rows=[{"name": record["id"], **record} for record in records],
        evaluate_fixture=evaluate_record,
        evaluate_gates=gates,
        result_metadata={
            "schema_version": "91-rank-only-condition-outcome",
            "experiment": "v91_fresh_external_local_rank_only_shadow",
            "condition": model,
            "model_manifest_sha256": manifest["manifest_sha256"],
            "planner_invariance_result": lock["planner_invariance_result"],
            "planner_invariance_result_sha256": lock[
                "planner_invariance_result_sha256"
            ],
            "claim_boundary": (
                "offline rank-only shadow evaluation; deterministic completion retains every "
                "schema intent and NONE; no pruning, state mutation, belief update, action "
                "selection, API, training, execution, service call, retry, or side effect"
            ),
        },
        pass_decision=(
            "freeze_and_preregister_bounded_complete_search_scheduling_study_only"
        ),
        fail_decision=(
            "freeze_nonqualifying_ranker_and_retain_deterministic_exhaustive_order"
        ),
    )
    access["elapsed_seconds"] = time.perf_counter() - start
    access["peak_active_memory_bytes"] = int(mx.get_peak_memory())
    result["metrics"] = gates.last_metrics
    result["planner_invariance"] = planner
    result["quality_gate_pass"] = gates.last_quality_pass
    result["completed_condition"] = bool(
        len(result["fixtures"]) == config["accessGates"]["requiredRecordCount"]
        and access["model_load_count"] == 1
        and access["model_generation_count"] == len(records)
    )
    result["final_access"] = access
    write_json(output_dir / "result.json", result)
    write_json(output_dir / "access.json", access)
    print(
        json.dumps(
            {
                "completed_condition": result["completed_condition"],
                "quality_gate_pass": result["quality_gate_pass"],
                "decision": result["decision"],
                "metrics": result["metrics"],
                "planner_invariance": planner,
                "gates": result["gates"],
                "access": access,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
