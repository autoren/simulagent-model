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
from v105_open_world_interface import validate_response
from v106_open_world_benchmark import split_development_records
from v163_deterministic_open_set_transfer_baselines import adapt_development_records
from v164_local_residual_open_set_transfer import (
    aggregate_residual_fixtures,
    evaluate_quality_and_access_gates,
    render_residual_prompt,
)


def payload_hash(value: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def read_jsonl_bytes(value: bytes) -> list[dict[str, Any]]:
    return [json.loads(line) for line in value.decode().splitlines() if line]


def prepare_records(
    lock: dict[str, Any],
) -> tuple[
    list[dict[str, Any]], list[dict[str, Any]], dict[str, dict[str, Any]]
]:
    development_bytes = (PROJECT_ROOT / lock["development_language"]).read_bytes()
    if hashlib.sha256(development_bytes).hexdigest() != lock[
        "development_language_sha256"
    ]:
        raise RuntimeError("V164 development language identity mismatch")
    development = adapt_development_records(read_jsonl_bytes(development_bytes))
    split = split_development_records(development, lock["baseline_config_payload"])
    evaluation = split["evaluation"]
    residual_payload = json.loads(
        (PROJECT_ROOT / lock["residual_manifest"]).read_text()
    )
    residual_ids = {row["record_id"] for row in residual_payload["records"]}
    if (
        residual_payload["payload_sha256"]
        != lock["config_payload"]["corpus"]["residualManifestPayloadSha256"]
        or len(residual_ids)
        != lock["config_payload"]["corpus"]["modelEligibleResidualCount"]
    ):
        raise RuntimeError("V164 residual manifest mismatch")
    residual = [row for row in evaluation if row["record_id"] in residual_ids]
    if len(residual) != len(residual_ids):
        raise RuntimeError("V164 residual records are not an evaluation subset")
    prediction_payload = json.loads(
        (PROJECT_ROOT / lock["baseline_predictions"]).read_text()
    )
    consensus = prediction_payload["predictions"]["deterministic_consensus"]
    if set(consensus) != {row["record_id"] for row in evaluation}:
        raise RuntimeError("V164 deterministic prediction identity mismatch")
    residual.sort(key=lambda row: row["record_id"])
    return residual, evaluation, consensus


def main() -> None:
    lock_path = (
        PROJECT_ROOT / "configs/v164-local-residual-open-set-transfer-lock.json"
    )
    lock = json.loads(lock_path.read_text())
    if payload_hash(
        {key: value for key, value in lock.items() if key != "lock_payload_sha256"}
    ) != lock["lock_payload_sha256"]:
        raise RuntimeError("V164 implementation lock mismatch")
    dependency_keys = (
        "config",
        "parent_deterministic_outcome",
        "parent_baseline_lock",
        "historical_interface_outcome",
        "historical_interface_lock",
        "direct_decoding_evidence",
        "visible_catalog",
        "safe_hypothesis_universe",
        "residual_manifest",
        "baseline_predictions",
        "model_manifest",
        "plan",
        "protocol",
        "tests",
        "runner",
        "verifier",
        "auditor",
        "census_harness",
        "implementation_audit",
    )
    for key in dependency_keys:
        if file_sha256(PROJECT_ROOT / lock[key]) != lock[f"{key}_sha256"]:
            raise RuntimeError(f"V164 dependency drifted: {key}")
    config = lock["config_payload"]
    catalog = json.loads((PROJECT_ROOT / lock["visible_catalog"]).read_text())
    manifest = json.loads((PROJECT_ROOT / lock["model_manifest"]).read_text())
    snapshot = Path(manifest["snapshot_path"])
    if not snapshot.is_dir() or snapshot.name != config["condition"]["revision"]:
        raise RuntimeError("V164 pinned snapshot is unavailable")
    residual_records, evaluation_records, consensus_predictions = prepare_records(lock)
    if len(residual_records) != config["corpus"]["totalModelGenerationCount"]:
        raise RuntimeError("V164 residual generation count mismatch")
    parent = json.loads(
        (PROJECT_ROOT / lock["parent_deterministic_outcome"]).read_text()
    )
    frozen_consensus_regret = parent["outcome"]["baseline_metrics"][
        "deterministic_consensus"
    ]["mean_regret"]
    output_dir = (
        PROJECT_ROOT / "outputs/v164-local-residual-open-set-transfer/development"
    )
    access: dict[str, Any] = {
        "condition_id": config["condition"]["id"],
        "development_language_read_count": 1,
        "protected_language_read_count": 0,
        "manual_utterance_inspection_count": 0,
        "manual_raw_response_inspection_count": 0,
        "model_load_count": 0,
        "model_generation_count": 0,
        "retry_count": 0,
        "LLM_API_call_count": 0,
        "adapter_training_run_count": 0,
        "real_service_call_count": 0,
        "external_side_effect_count": 0,
        "actual_execution_count": 0,
    }
    state: dict[str, Any] = {"model": None, "tokenizer": None}
    mx.reset_peak_memory()
    condition_start = time.perf_counter()

    fixture_rows = [
        {"name": row["record_id"], "record": row} for row in residual_records
    ]

    def evaluate_fixture(fixture: dict[str, Any]) -> dict[str, Any]:
        if state["model"] is None:
            load_start = time.perf_counter()
            model, tokenizer = load(str(snapshot))
            model.eval()
            state["model"] = model
            state["tokenizer"] = tokenizer
            access["model_load_count"] += 1
            access["model_load_seconds"] = time.perf_counter() - load_start
        user_payload = render_residual_prompt(
            catalog, fixture["record"]["utterance"], config
        )
        tokenizer = state["tokenizer"]
        prompt = tokenizer.apply_chat_template(
            [
                {"role": "system", "content": config["prompt"]["system"]},
                {"role": "user", "content": user_payload},
            ],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        prompt_tokens = tokenizer.encode(prompt)
        if len(prompt_tokens) > config["prompt"]["maximumPromptTokens"]:
            raise RuntimeError("V164 prompt exceeds locked token budget")
        generation_start = time.perf_counter()
        raw_response = generate(
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
        parsed, valid, reason = validate_response(
            raw_response, catalog, lock["interface_config_payload"]
        )
        return {
            "name": fixture["name"],
            "raw_response": raw_response,
            "raw_response_sha256": hashlib.sha256(raw_response.encode()).hexdigest(),
            "parsed_response": parsed,
            "response_valid": valid,
            "validation_reason": reason,
            "prompt_token_count": len(prompt_tokens),
            "generated_token_count": len(tokenizer.encode(raw_response)),
            "generation_seconds": generation_seconds,
            "permanently_non_authoritative": True,
            "safe_hypothesis_universe_pruned": False,
            "nonresidual_decision_overridden": False,
            "executable": False,
        }

    def gates(fixtures: dict[str, dict[str, Any]]) -> dict[str, bool]:
        aggregate = aggregate_residual_fixtures(
            fixtures,
            residual_records,
            evaluation_records,
            consensus_predictions,
            lock["baseline_config_payload"],
        )
        gates.last_aggregate = aggregate
        return evaluate_quality_and_access_gates(
            aggregate, frozen_consensus_regret, access, config
        )

    gates.last_aggregate = None
    result = run_locked_census_once(
        output_dir=output_dir,
        attempt=access,
        fixture_rows=fixture_rows,
        evaluate_fixture=evaluate_fixture,
        evaluate_gates=gates,
        result_metadata={
            "schema_version": "164-local-residual-open-set-transfer-result",
            "experiment": config["experiment"],
            "condition": config["condition"],
            "model_manifest_sha256": manifest["manifest_sha256"],
            "frozen_consensus_regret": frozen_consensus_regret,
            "claim_boundary": config["claimBoundary"],
        },
        pass_decision=(
            "local_residual_hybrid_qualifies_for_separate_protected_protocol_preregistration_only"
        ),
        fail_decision=(
            "local_residual_hybrid_is_nonqualifying_and_protected_transfer_remains_sealed"
        ),
    )
    access["elapsed_seconds"] = time.perf_counter() - condition_start
    access["peak_active_memory_bytes"] = int(mx.get_peak_memory())
    result["aggregate"] = gates.last_aggregate
    result["completed_condition"] = bool(
        len(result["fixtures"]) == config["accessGates"]["requiredResidualFixtureCount"]
        and access["model_load_count"] == 1
        and access["model_generation_count"]
        == config["corpus"]["totalModelGenerationCount"]
    )
    result["final_access"] = access
    write_json(output_dir / "result.json", result)
    write_json(output_dir / "access.json", access)
    print(
        json.dumps(
            {
                "completed_condition": result["completed_condition"],
                "passed": result["passed"],
                "decision": result["decision"],
                "aggregate": result["aggregate"],
                "gates": result["gates"],
                "access": access,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
