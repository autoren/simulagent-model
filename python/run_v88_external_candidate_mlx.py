#!/usr/bin/env python3
"""Run the one frozen local-only V88 external-language shadow census."""
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


def main() -> None:
    lock_path = PROJECT_ROOT / "configs/v88-external-intent-candidate-implementation-lock.json"
    lock = json.loads(lock_path.read_text())
    payload = {key: value for key, value in lock.items() if key != "lock_payload_sha256"}
    if payload_hash(payload) != lock["lock_payload_sha256"]:
        raise RuntimeError("V88 implementation lock drifted")
    if not lock["authorization"]["run_local_model_once"]:
        raise RuntimeError("V88 implementation lock does not authorize inference")
    for path_key, hash_key in (
        ("design_lock", "design_lock_sha256"),
        ("corpus_seal", "corpus_seal_sha256"),
        ("corpus", "corpus_sha256"),
        ("protocol", "protocol_sha256"),
        ("runner", "runner_sha256"),
        ("census_harness", "census_harness_sha256"),
    ):
        if file_sha256(PROJECT_ROOT / lock[path_key]) != lock[hash_key]:
            raise RuntimeError(f"V88 locked dependency drifted: {path_key}")
    snapshot = Path(lock["local_snapshot_path"])
    if not snapshot.is_dir():
        raise RuntimeError("V88 pinned local snapshot is unavailable")

    config = lock["config_payload"]
    records = read_jsonl(PROJECT_ROOT / lock["corpus"])
    records_by_id = {record["id"]: record for record in records}
    output_dir = PROJECT_ROOT / "outputs/v88-external-intent-candidate/evaluation"
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
            raise RuntimeError("V88 prompt exceeds the locked budget")
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
        row = score_response(record, response)
        row["prompt_token_count"] = len(prompt_tokens)
        return row

    def gates(fixtures: dict) -> dict[str, bool]:
        metrics = aggregate(list(fixtures.values()), records_by_id)
        gates.last_metrics = metrics
        return evaluate_gates(metrics, config, access)

    gates.last_metrics = None
    result = run_locked_census_once(
        output_dir=output_dir,
        attempt=access,
        fixture_rows=[{"name": record["id"], **record} for record in records],
        evaluate_fixture=evaluate_record,
        evaluate_gates=gates,
        result_metadata={
            "schema_version": "88-external-intent-candidate-outcome",
            "experiment": "v88_sealed_external_human_language_local_candidate_shadow",
            "model": config["model"],
            "source": config["source"],
            "claim_boundary": (
                "offline external human-language shadow candidate proposal only; every output "
                "permanently non-deployable; no API, training, manual inspection, service call, "
                "action selection, belief authority, or side effect"
            ),
        },
        pass_decision="freeze_positive_V88_external_human_language_shadow_outcome",
        fail_decision="freeze_negative_V88_without_prompt_record_gate_or_model_changes",
    )
    result["metrics"] = gates.last_metrics
    write_json(output_dir / "result.json", result)
    write_json(output_dir / "access.json", access)
    print(json.dumps({
        "passed": result["passed"],
        "decision": result["decision"],
        "metrics": result["metrics"],
        "gates": result["gates"],
        "access": access,
    }, indent=2, sort_keys=True))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
