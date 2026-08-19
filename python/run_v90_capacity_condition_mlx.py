#!/usr/bin/env python3
"""Run one independently locked V90 local-model condition exactly once."""
from __future__ import annotations

import argparse
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
from v90_capacity_generation_protocol import (
    aggregate,
    evaluate_condition_gates,
    format_user_prompt,
    quality_gate_pass,
    score_response,
)


def payload_hash(value: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def score_named_record(record: dict[str, Any], response: str) -> dict[str, Any]:
    row = score_response(record, response)
    row["name"] = record["name"]
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition", required=True)
    args = parser.parse_args()

    lock_path = PROJECT_ROOT / "configs/v90-capacity-generation-implementation-lock.json"
    lock = json.loads(lock_path.read_text())
    payload = {key: value for key, value in lock.items() if key != "lock_payload_sha256"}
    if payload_hash(payload) != lock["lock_payload_sha256"]:
        raise RuntimeError("V90 implementation lock payload mismatch")
    for key in ("design_lock", "corpus_seal", "corpus", "protocol", "tests", "runner", "census_harness", "acquisition_result", "implementation_auditor"):
        if file_sha256(PROJECT_ROOT / lock[key]) != lock[f"{key}_sha256"]:
            raise RuntimeError(f"V90 implementation dependency drifted: {key}")
    conditions = {item["id"]: item for item in lock["config_payload"]["modelConditions"]}
    if args.condition not in conditions:
        raise RuntimeError("condition is not registered in the V90 implementation lock")
    if args.condition not in lock["authorization"]["run_each_registered_condition_once"]:
        raise RuntimeError("condition is not authorized for V90 inference")
    condition = conditions[args.condition]
    manifest_spec = lock["model_manifests"][args.condition]
    manifest_path = PROJECT_ROOT / manifest_spec["path"]
    if file_sha256(manifest_path) != manifest_spec["sha256"]:
        raise RuntimeError("V90 model manifest drifted")
    manifest = json.loads(manifest_path.read_text())
    snapshot = Path(manifest["snapshot_path"])
    if not snapshot.is_dir() or snapshot.name != condition["revision"]:
        raise RuntimeError("V90 pinned snapshot is unavailable")

    config = lock["config_payload"]
    records = read_jsonl(PROJECT_ROOT / lock["corpus"])
    records_by_id = {record["id"]: record for record in records}
    output_dir = PROJECT_ROOT / "outputs/v90-capacity-generation/evaluation" / args.condition
    access = {
        "condition_id": args.condition,
        "source_language_record_access_count": len(records),
        "manual_utterance_inspection_count": 0,
        "model_load_count": 0,
        "model_generation_count": 0,
        "LLM_API_call_count": 0,
        "adapter_training_run_count": 0,
        "real_service_call_count": 0,
        "external_side_effect_count": 0,
    }
    state: dict[str, Any] = {"model": None, "tokenizer": None}
    mx.reset_peak_memory()
    condition_start = time.perf_counter()

    def evaluate_record(record: dict[str, Any]) -> dict[str, Any]:
        if state["model"] is None:
            load_start = time.perf_counter()
            model, tokenizer = load(str(snapshot))
            model.eval()
            state["model"] = model
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
            raise RuntimeError("V90 prompt exceeds locked token budget")
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
        access["elapsed_seconds"] = time.perf_counter() - condition_start
        access["peak_active_memory_bytes"] = int(mx.get_peak_memory())
        write_json(output_dir / "access-progress.json", access)
        row = score_named_record(record, response)
        row.update({
            "condition_id": args.condition,
            "prompt_token_count": len(prompt_tokens),
            "generated_token_count": len(tokenizer.encode(response)),
            "generation_seconds": generation_seconds,
            "permanently_non_deployable": True,
            "executable": False,
        })
        return row

    def gates(fixtures: dict[str, dict[str, Any]]) -> dict[str, bool]:
        metrics = aggregate(list(fixtures.values()), records_by_id)
        gates.last_metrics = metrics
        checks = evaluate_condition_gates(metrics, config, access)
        gates.last_quality_pass = quality_gate_pass(checks)
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
            "schema_version": "90-capacity-generation-condition-outcome",
            "experiment": "v90_fresh_external_local_capacity_generation_shadow",
            "condition": condition,
            "model_manifest_sha256": manifest["manifest_sha256"],
            "claim_boundary": (
                "offline human-language shadow candidate proposal only; every output is permanently "
                "non-deployable and non-executable; no API, training, manual inspection, belief authority, "
                "action selection, service call, retry, or side effect"
            ),
        },
        pass_decision="condition_qualifies_for_separately_preregistered_posterior_integration_shadow_only",
        fail_decision="condition_does_not_qualify_and_remains_frozen_shadow_evidence",
    )
    access["elapsed_seconds"] = time.perf_counter() - condition_start
    access["peak_active_memory_bytes"] = int(mx.get_peak_memory())
    result["metrics"] = gates.last_metrics
    result["quality_gate_pass"] = gates.last_quality_pass
    result["completed_condition"] = bool(
        len(result["fixtures"]) == config["accessGatesPerCondition"]["requiredRecordCount"]
        and access["model_load_count"] == 1
        and access["model_generation_count"] == len(records)
    )
    result["final_access"] = access
    write_json(output_dir / "result.json", result)
    write_json(output_dir / "access.json", access)
    print(json.dumps({
        "condition_id": args.condition,
        "completed_condition": result["completed_condition"],
        "quality_gate_pass": result["quality_gate_pass"],
        "decision": result["decision"],
        "metrics": result["metrics"],
        "gates": result["gates"],
        "access": access,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
