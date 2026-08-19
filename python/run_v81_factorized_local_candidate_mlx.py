#!/usr/bin/env python3
"""Run the one locked V81 factorized local-only evaluation."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from mlx_lm import generate, load
from mlx_lm.sample_utils import make_sampler

from locked_census_harness import run_locked_census_once, write_json
from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT
from v81_factorized_candidate_protocol import aggregate, evaluate_gates, score_record


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def main() -> None:
    lock_path = (
        PROJECT_ROOT / "configs/v81-factorized-local-candidate-implementation-lock.json"
    )
    lock = json.loads(lock_path.read_text())
    payload = {key: value for key, value in lock.items() if key != "lock_payload_sha256"}
    if hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest() != lock["lock_payload_sha256"]:
        raise RuntimeError("V81 implementation lock payload drifted")
    if not lock["authorization"]["run_local_model_once"]:
        raise RuntimeError("V81 lock does not authorize local inference")
    for path_key, hash_key in (
        ("design_lock", "design_lock_sha256"),
        ("corpus_seal", "corpus_seal_sha256"),
        ("corpus", "corpus_sha256"),
        ("protocol", "protocol_sha256"),
        ("runner", "runner_sha256"),
        ("census_harness", "census_harness_sha256"),
    ):
        if file_sha256(PROJECT_ROOT / lock[path_key]) != lock[hash_key]:
            raise RuntimeError(f"V81 locked dependency drifted: {path_key}")

    snapshot = Path(lock["local_snapshot_path"])
    if not snapshot.is_dir():
        raise RuntimeError("V81 pinned local snapshot is unavailable")
    config = lock["config_payload"]
    records = read_jsonl(PROJECT_ROOT / lock["corpus"])
    output_dir = PROJECT_ROOT / "outputs/v81-factorized-local-candidate/evaluation"
    access = {
        "attempt_number": 1,
        "model_load_count": 0,
        "model_generation_count": 0,
        "API_call_count": 0,
        "adapter_training_run_count": 0,
        "human_record_access_count": 0,
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
                        instruction=record["instruction"]
                    ),
                },
            ],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        prompt_tokens = tokenizer.encode(prompt)
        if len(prompt_tokens) > config["decoding"]["maximumPromptTokens"]:
            raise RuntimeError("V81 prompt exceeds its locked token budget")
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
        row = score_record(record, response, config)
        row["name"] = record["id"]
        row["prompt_token_count"] = len(prompt_tokens)
        return row

    def gates(fixtures: dict) -> dict[str, bool]:
        metrics = aggregate(list(fixtures.values()))
        gates.last_metrics = metrics
        return evaluate_gates(metrics, config, access)

    gates.last_metrics = None
    result = run_locked_census_once(
        output_dir=output_dir,
        attempt=access,
        fixture_rows=[{"name": row["id"], **row} for row in records],
        evaluate_fixture=evaluate_record,
        evaluate_gates=gates,
        result_metadata={
            "schema_version": "81-factorized-local-candidate-outcome",
            "experiment": "v81_frozen_local_LLM_factorized_candidate_generation",
            "model": config["model"],
            "claim_boundary": (
                "frozen local model emitted fixed Boolean semantic labels; deterministic "
                "code composed candidates; no calibrated beliefs, decision authority, "
                "API, adapter, human evidence, tool execution, or side effect"
            ),
        },
        pass_decision=(
            "freeze_factorized_proposer_and_authorize_candidate_to_belief_preregistration"
        ),
        fail_decision=(
            "freeze_V81_failure_without_edits_or_rerun_and_stop_local_candidate_integration"
        ),
    )
    result["metrics"] = gates.last_metrics
    write_json(output_dir / "result.json", result)
    write_json(output_dir / "access.json", access)
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
