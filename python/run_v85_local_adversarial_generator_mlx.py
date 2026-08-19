#!/usr/bin/env python3
"""Run the one frozen local-only V85 adversarial generation census."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from mlx_lm import generate, load
from mlx_lm.sample_utils import make_sampler

from locked_census_harness import run_locked_census_once, write_json
from schema_grounded_interface import compile_schema_registry, unsafe_schema_surface_mutations
from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT
from v85_adversarial_protocol import aggregate, evaluate_gates, score_response


def payload_hash(value: dict) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def main() -> None:
    lock_path = PROJECT_ROOT / "configs/v85-local-adversarial-generator-implementation-lock.json"
    lock = json.loads(lock_path.read_text())
    payload = {key: value for key, value in lock.items() if key != "lock_payload_sha256"}
    if payload_hash(payload) != lock["lock_payload_sha256"]:
        raise RuntimeError("V85 implementation lock drifted")
    if not lock["authorization"]["run_local_model_once"]:
        raise RuntimeError("V85 implementation lock does not authorize inference")
    for path_key, hash_key in (
        ("design_lock", "design_lock_sha256"),
        ("corpus_seal", "corpus_seal_sha256"),
        ("corpus", "corpus_sha256"),
        ("protocol", "protocol_sha256"),
        ("runner", "runner_sha256"),
        ("census_harness", "census_harness_sha256"),
        ("schema_source_lock", "schema_source_lock_sha256"),
    ):
        if file_sha256(PROJECT_ROOT / lock[path_key]) != lock[hash_key]:
            raise RuntimeError(f"V85 locked dependency drifted: {path_key}")
    snapshot = Path(lock["local_snapshot_path"])
    if not snapshot.is_dir():
        raise RuntimeError("V85 pinned local snapshot is unavailable")
    config = lock["config_payload"]
    records = read_jsonl(PROJECT_ROOT / lock["corpus"])
    registry = compile_schema_registry(lock["schemas"])
    schema_by_id = {schema.schema_id: schema for schema in registry.schemas}
    deterministic = {question for _, question in unsafe_schema_surface_mutations(registry)}
    output_dir = PROJECT_ROOT / "outputs/v85-local-adversarial-generator/evaluation"
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
        schema = schema_by_id[record["schemaId"]]
        slot1, slot2 = schema.slots
        user_prompt = config["userPromptTemplate"].format(
            schemaId=record["schemaId"],
            kind=record["kind"],
            slotId=record["slotId"] if record["slotId"] is not None else "null",
            profile=record["profile"],
            slot1Prefix=slot1.question_prefix,
            slot1Choice1=slot1.options[0].surface,
            slot1Choice2=slot1.options[1].surface,
            slot2Prefix=slot2.question_prefix,
            slot2Choice1=slot2.options[0].surface,
            slot2Choice2=slot2.options[1].surface,
        )
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
            raise RuntimeError("V85 prompt exceeds the locked budget")
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
        row = score_response(record, response, registry, config, deterministic)
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
        fixture_rows=[{"name": record["id"], **record} for record in records],
        evaluate_fixture=evaluate_record,
        evaluate_gates=gates,
        result_metadata={
            "schema_version": "85-local-adversarial-generator-outcome",
            "experiment": "v85_offline_non_deployable_local_adversarial_surface_generation",
            "model": config["model"],
            "claim_boundary": (
                "offline local adversarial test-input generation only; every output permanently "
                "non-deployable; no original language, schema/belief/action authority, API, "
                "training, human record, tool, or side effect"
            ),
        },
        pass_decision="freeze_positive_offline_local_adversarial_generator_outcome",
        fail_decision="freeze_negative_V85_without_prompt_edit_or_rerun",
    )
    result["metrics"] = gates.last_metrics
    write_json(output_dir / "result.json", result)
    write_json(output_dir / "access.json", access)
    print(json.dumps({"passed": result["passed"], "decision": result["decision"], "metrics": result["metrics"], "gates": result["gates"]}, indent=2, sort_keys=True))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
