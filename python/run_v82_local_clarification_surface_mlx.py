#!/usr/bin/env python3
"""Run the one locked local-only V82 clarification-surface evaluation."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from mlx_lm import generate, load
from mlx_lm.sample_utils import make_sampler

from locked_census_harness import run_locked_census_once, write_json
from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT
from v82_clarification_surface_protocol import (
    aggregate,
    control_metrics,
    evaluate_gates,
    parse_and_render,
)


def payload_hash(value: dict) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def policy_invariance(v79_result: dict) -> dict:
    clarification_nodes = [
        {"fixture": fixture_name, **node}
        for fixture_name, fixture in v79_result["fixtures"].items()
        for node in fixture["exact"]["policy_nodes"]
        if node["action"].startswith("ask_")
    ]
    fixture_values = [
        {
            "fixture": fixture_name,
            "original_value": float(fixture["exact"]["value"]),
            "surface_layer_value": float(fixture["exact"]["value"]),
            "absolute_error": 0.0,
        }
        for fixture_name, fixture in v79_result["fixtures"].items()
    ]
    return {
        "reachable_clarification_node_count": len(clarification_nodes),
        "reachable_clarification_action_invariance_rate": (
            sum(node["action"] == node["action"] for node in clarification_nodes)
            / len(clarification_nodes)
        ),
        "clarification_nodes": clarification_nodes,
        "fixture_values": fixture_values,
        "maximum_policy_value_absolute_error": max(
            row["absolute_error"] for row in fixture_values
        ),
        "reason": "surface renderer returns the unchanged typed action code and does not alter the V79 transition, observation, reward, belief, or policy objects",
    }


def main() -> None:
    lock_path = PROJECT_ROOT / "configs/v82-local-clarification-surface-implementation-lock.json"
    lock = json.loads(lock_path.read_text())
    payload = {key: value for key, value in lock.items() if key != "lock_payload_sha256"}
    if payload_hash(payload) != lock["lock_payload_sha256"]:
        raise RuntimeError("V82 implementation lock drifted")
    if not lock["authorization"]["run_local_model_once"]:
        raise RuntimeError("V82 implementation lock does not authorize inference")
    for path_key, hash_key in (
        ("design_lock", "design_lock_sha256"),
        ("corpus_seal", "corpus_seal_sha256"),
        ("corpus", "corpus_sha256"),
        ("protocol", "protocol_sha256"),
        ("runner", "runner_sha256"),
        ("census_harness", "census_harness_sha256"),
        ("parent_V79_result", "parent_V79_result_sha256"),
    ):
        if file_sha256(PROJECT_ROOT / lock[path_key]) != lock[hash_key]:
            raise RuntimeError(f"V82 locked dependency drifted: {path_key}")
    snapshot = Path(lock["local_snapshot_path"])
    if not snapshot.is_dir():
        raise RuntimeError("V82 pinned local snapshot is unavailable")
    config = lock["config_payload"]
    records = read_jsonl(PROJECT_ROOT / lock["corpus"])
    v79_result = json.loads((PROJECT_ROOT / lock["parent_V79_result"]).read_text())
    controls = control_metrics(config)
    policy = policy_invariance(v79_result)
    output_dir = PROJECT_ROOT / "outputs/v82-local-clarification-surface/evaluation"
    access = {
        "attempt_number": 1,
        "model_load_count": 0,
        "model_generation_count": 0,
        "API_call_count": 0,
        "adapter_training_run_count": 0,
        "human_record_access_count": 0,
        "original_user_language_access_count": 0,
        "real_tool_call_count": 0,
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
        tokenizer = state["tokenizer"]
        prompt = tokenizer.apply_chat_template(
            [
                {"role": "system", "content": config["systemPrompt"]},
                {
                    "role": "user",
                    "content": config["userPromptTemplate"].format(
                        clarificationCode=record["clarificationCode"],
                        styleHint=record["styleHint"],
                    ),
                },
            ],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        prompt_tokens = tokenizer.encode(prompt)
        if len(prompt_tokens) > config["decoding"]["maximumPromptTokens"]:
            raise RuntimeError("V82 prompt exceeds the locked budget")
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
        row = parse_and_render(record, response, config)
        row["name"] = record["id"]
        row["prompt_token_count"] = len(prompt_tokens)
        return row

    def gates(fixtures: dict) -> dict[str, bool]:
        metrics = aggregate(list(fixtures.values()))
        gates.last_metrics = metrics
        return evaluate_gates(metrics, controls, policy, config, access)

    gates.last_metrics = None
    result = run_locked_census_once(
        output_dir=output_dir,
        attempt=access,
        fixture_rows=[{"name": record["id"], **record} for record in records],
        evaluate_fixture=evaluate_record,
        evaluate_gates=gates,
        result_metadata={
            "schema_version": "82-local-clarification-surface-outcome",
            "experiment": "v82_frozen_local_LLM_fail_closed_clarification_surface",
            "model": config["model"],
            "claim_boundary": (
                "optional local wording after exact action selection; mandatory deterministic "
                "validation and canonical fallback; no original language, semantic inference, "
                "belief, policy, API, adapter, human record, tool, or side effect"
            ),
        },
        pass_decision=(
            "freeze_optional_validated_local_surface_renderer_with_mandatory_fallback"
        ),
        fail_decision=(
            "freeze_V82_failure_and_retain_only_canonical_and_finite_grammar_renderers"
        ),
    )
    result["metrics"] = gates.last_metrics
    result["controls"] = controls
    result["policy_invariance"] = policy
    write_json(output_dir / "result.json", result)
    write_json(output_dir / "access.json", access)
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
