#!/usr/bin/env python3
"""Run the single name-preserving mechanical retry of the frozen V88 census."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from mlx_lm import generate, load
from mlx_lm.sample_utils import make_sampler

from locked_census_harness import run_locked_census_once, write_json
from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT
from v88_external_candidate_protocol import aggregate, evaluate_gates, format_user_prompt, score_response


def payload_hash(value: dict) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def score_named_record(record: dict, response: str) -> dict:
    """The only V88r1 behavior change: preserve the harness fixture identity."""
    row = score_response(record, response)
    row["name"] = record["name"]
    return row


def main() -> None:
    lock_path = PROJECT_ROOT / "configs/v88r1-name-preservation-repair-implementation-lock.json"
    lock = json.loads(lock_path.read_text())
    payload = {key: value for key, value in lock.items() if key != "lock_payload_sha256"}
    if payload_hash(payload) != lock["lock_payload_sha256"]:
        raise RuntimeError("V88r1 implementation lock drifted")
    if not lock["authorization"]["run_single_local_repair_once"]:
        raise RuntimeError("V88r1 implementation lock does not authorize inference")
    for path_key, hash_key in (
        ("repair_design_lock", "repair_design_lock_sha256"),
        ("original_implementation_lock", "original_implementation_lock_sha256"),
        ("corpus_seal", "corpus_seal_sha256"),
        ("corpus", "corpus_sha256"),
        ("protocol", "protocol_sha256"),
        ("runner", "runner_sha256"),
        ("census_harness", "census_harness_sha256"),
    ):
        if file_sha256(PROJECT_ROOT / lock[path_key]) != lock[hash_key]:
            raise RuntimeError(f"V88r1 locked dependency drifted: {path_key}")
    snapshot = Path(lock["local_snapshot_path"])
    if not snapshot.is_dir():
        raise RuntimeError("V88r1 pinned local snapshot is unavailable")

    config = lock["config_payload"]
    records = read_jsonl(PROJECT_ROOT / lock["corpus"])
    records_by_id = {record["id"]: record for record in records}
    prior = lock["prior_failed_attempt_access"]
    cumulative_budget = lock["cumulative_resource_budget"]
    output_dir = PROJECT_ROOT / "outputs/v88r1-external-intent-candidate/evaluation"
    access = {
        "attempt_number": 1,
        "source_language_record_access_count": len(records),
        "manual_utterance_inspection_count": 0,
        "model_load_count": 0,
        "model_generation_count": 0,
        "LLM_API_call_count": 0,
        "adapter_training_run_count": 0,
        "real_service_call_count": 0,
        "external_side_effect_count": 0,
    }
    state = {"model": None, "tokenizer": None}

    def evaluate_record(record: dict) -> dict:
        if state["model"] is None:
            model, tokenizer = load(str(snapshot))
            model.eval()
            state["model"] = model
            state["tokenizer"] = tokenizer
            access["model_load_count"] += 1
        user_prompt = format_user_prompt(record, config)
        tokenizer = state["tokenizer"]
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
            raise RuntimeError("V88r1 prompt exceeds the unchanged locked budget")
        response = generate(
            state["model"],
            tokenizer,
            prompt=prompt,
            max_tokens=config["decoding"]["maximumNewTokens"],
            sampler=make_sampler(temp=config["decoding"]["temperature"]),
            verbose=False,
        )
        access["model_generation_count"] += 1
        write_json(output_dir / "access-progress.json", access)
        row = score_named_record(record, response)
        row["prompt_token_count"] = len(prompt_tokens)
        return row

    def gates(fixtures: dict) -> dict[str, bool]:
        metrics = aggregate(list(fixtures.values()), records_by_id)
        gates.last_metrics = metrics
        checks = evaluate_gates(metrics, config, access)
        checks["cumulative_model_load_budget"] = (
            prior["model_load_count"] + access["model_load_count"]
            <= cumulative_budget["maximum_model_load_count"]
        )
        checks["cumulative_model_generation_budget"] = (
            prior["model_generation_count"] + access["model_generation_count"]
            <= cumulative_budget["maximum_model_generation_count"]
        )
        return checks

    gates.last_metrics = None
    result = run_locked_census_once(
        output_dir=output_dir,
        attempt=access,
        fixture_rows=[{"name": record["id"], **record} for record in records],
        evaluate_fixture=evaluate_record,
        evaluate_gates=gates,
        result_metadata={
            "schema_version": "88r1-external-intent-candidate-outcome",
            "experiment": "v88r1_sealed_external_human_language_local_candidate_shadow",
            "mechanical_repair": "preserve registered harness fixture name only",
            "model": config["model"],
            "source": config["source"],
            "claim_boundary": (
                "offline external human-language shadow candidate proposal only; every output "
                "permanently non-deployable; one disclosed mechanical retry after an unscored harness "
                "identity failure; no API, training, manual inspection, service call, action selection, "
                "belief authority, further retry, or side effect"
            ),
        },
        pass_decision="freeze_positive_V88r1_external_human_language_shadow_outcome",
        fail_decision="freeze_negative_V88r1_without_any_further_retry_or_change",
    )
    result["metrics"] = gates.last_metrics
    result["prior_failed_attempt_access"] = prior
    result["cumulative_access"] = {
        "model_load_count": prior["model_load_count"] + access["model_load_count"],
        "model_generation_count": prior["model_generation_count"] + access["model_generation_count"],
        "LLM_API_call_count": prior["LLM_API_call_count"] + access["LLM_API_call_count"],
        "adapter_training_run_count": prior["adapter_training_run_count"] + access["adapter_training_run_count"],
        "manual_utterance_inspection_count": prior["manual_utterance_inspection_count"] + access["manual_utterance_inspection_count"],
        "real_service_call_count": prior["real_service_call_count"] + access["real_service_call_count"],
        "external_side_effect_count": prior["external_side_effect_count"] + access["external_side_effect_count"],
    }
    write_json(output_dir / "result.json", result)
    write_json(output_dir / "access.json", access)
    print(json.dumps({
        "passed": result["passed"],
        "decision": result["decision"],
        "metrics": result["metrics"],
        "gates": result["gates"],
        "retry_access": access,
        "cumulative_access": result["cumulative_access"],
    }, indent=2, sort_keys=True))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
