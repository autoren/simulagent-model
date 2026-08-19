#!/usr/bin/env python3
"""One-shot bounded local Phase 2 run for the prospective language pilot."""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

import mlx.core as mx
from mlx_lm import generate, load, stream_generate
from mlx_lm.sample_utils import make_sampler

from locked_census_harness import run_locked_census_once, write_json
from prospective_language_phase2 import (
    controller_output,
    parse_semantic_proposal,
    render_phase2_user_payload,
    validate_phase2_config,
)
from prospective_language_pilot import canonical_json, sha256_bytes, sha256_json
from v10_protocol import file_sha256
from v154_adaptive_local_question_order import prepare_bounded_final_prompt_tokens


ROOT = Path(__file__).resolve().parents[1]
DESIGN_PATH = ROOT / "configs" / "prospective-language-pilot-v1-phase2.json"
PARTICIPANT_DIR = (
    ROOT / "data" / "prospective-language-pilot" / "prospective-language-pilot-v1" / "P001"
)
LOCK_PATH = PARTICIPANT_DIR / "audit" / "phase2_run_lock.json"
OUTPUT_DIR = PARTICIPANT_DIR / "assistant" / "phase2_architecture"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def jsonl_bytes(records: list[dict[str, Any]]) -> bytes:
    return ("\n".join(canonical_json(record) for record in records) + "\n").encode("utf-8")


def write_bytes(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)


def valid_lock(lock: dict[str, Any]) -> bool:
    payload = {key: value for key, value in lock.items() if key != "lock_payload_sha256"}
    return lock.get("lock_payload_sha256") == sha256_json(payload)


def main() -> None:
    if not LOCK_PATH.is_file():
        raise RuntimeError("Phase 2 must be audited and locked before generation.")
    if OUTPUT_DIR.exists():
        raise RuntimeError("The exact Phase 2 architecture condition may run only once.")

    lock = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
    if not valid_lock(lock):
        raise RuntimeError("Phase 2 run lock payload mismatch.")
    for dependency in lock["dependencies"]:
        path = ROOT / dependency["path"]
        if file_sha256(path) != dependency["sha256"]:
            raise RuntimeError(f"Phase 2 dependency drifted: {dependency['path']}")
    authorization = lock["authorization"]
    if not (
        authorization["run_exact_single_bounded_local_architecture_condition"]
        and not authorization["retry_reprompt_or_generate_an_alternate_condition"]
        and not authorization["use_private_scenario_cards_or_future_answers"]
        and not authorization["use_api_training_authority_action_or_execution"]
    ):
        raise RuntimeError("Phase 2 authorization is invalid.")

    config = json.loads(DESIGN_PATH.read_text(encoding="utf-8"))
    validate_phase2_config(config)
    public_path = ROOT / config["participant"]["public_requests"]
    if file_sha256(public_path) != config["participant"]["public_requests_sha256"]:
        raise RuntimeError("Locked Phase 1 public requests drifted.")
    records = read_jsonl(public_path)
    if (
        len(records) != config["participant"]["required_record_count"]
        or any(row["participant_code"] != config["participant"]["participant_code"] for row in records)
    ):
        raise RuntimeError("Locked Phase 1 record population mismatch.")

    manifest = json.loads((ROOT / config["model"]["model_manifest"]).read_text(encoding="utf-8"))
    snapshot = Path(manifest["snapshot_path"])
    if (
        manifest["repository"] != config["model"]["repository"]
        or manifest["revision"] != config["model"]["revision"]
        or not snapshot.is_dir()
    ):
        raise RuntimeError("Pinned local model snapshot mismatch.")

    access: dict[str, Any] = {
        "tokenizer_load_count": 0,
        "model_load_count": 0,
        "reasoning_phase_generation_count": 0,
        "final_phase_generation_count": 0,
        "total_generation_count": 0,
        "maximum_generation_calls_per_request": 0,
        "retry_count": 0,
        "manual_reasoning_inspection_count": 0,
        "persisted_reasoning_count": 0,
        "api_call_count": 0,
        "training_run_count": 0,
        "real_service_call_count": 0,
        "external_side_effect_count": 0,
        "actual_execution_count": 0,
    }
    state: dict[str, Any] = {"model": None, "tokenizer": None}
    sampler = make_sampler(temp=config["model"]["temperature"])
    started = time.perf_counter()
    mx.reset_peak_memory()

    def persist_access() -> None:
        access["total_generation_count"] = (
            access["reasoning_phase_generation_count"] + access["final_phase_generation_count"]
        )
        access["elapsed_seconds"] = time.perf_counter() - started
        access["peak_active_memory_bytes"] = int(mx.get_peak_memory())
        write_json(OUTPUT_DIR / "access-progress.json", access)

    def ensure_loaded() -> None:
        if state["model"] is not None:
            return
        load_started = time.perf_counter()
        model, tokenizer = load(str(snapshot))
        model.eval()
        state["model"], state["tokenizer"] = model, tokenizer
        access["model_load_count"] += 1
        access["tokenizer_load_count"] += 1
        access["model_load_seconds"] = time.perf_counter() - load_started
        persist_access()

    def evaluate_record(wrapper: dict[str, Any]) -> dict[str, Any]:
        ensure_loaded()
        record = wrapper["record"]
        payload = render_phase2_user_payload(record, config)
        messages = [
            {"role": "system", "content": config["prompt"]["system"]},
            {"role": "user", "content": payload},
        ]
        prompt = state["tokenizer"].apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,
            reasoning_effort=config["model"]["reasoning_effort"],
        )
        prompt_tokens = list(state["tokenizer"].encode(prompt, add_special_tokens=False))
        if len(prompt_tokens) > config["prompt"]["maximum_prompt_tokens"]:
            raise RuntimeError(f"Prompt exceeds frozen budget for {record['record_id']}.")

        access["reasoning_phase_generation_count"] += 1
        access["maximum_generation_calls_per_request"] = max(
            access["maximum_generation_calls_per_request"], 2
        )
        persist_access()
        generation_started = time.perf_counter()
        responses = list(
            stream_generate(
                state["model"],
                state["tokenizer"],
                prompt=prompt_tokens,
                max_tokens=config["model"]["reasoning_phase_maximum_tokens"],
                sampler=sampler,
            )
        )
        reasoning_tokens = [response.token for response in responses]
        reasoning_text = state["tokenizer"].decode(reasoning_tokens)
        final_prompt_tokens, natural_close, retained_count = prepare_bounded_final_prompt_tokens(
            prompt_tokens, reasoning_tokens, state["tokenizer"]
        )

        access["final_phase_generation_count"] += 1
        persist_access()
        raw = generate(
            state["model"],
            state["tokenizer"],
            prompt=final_prompt_tokens,
            max_tokens=config["model"]["final_phase_maximum_tokens"],
            sampler=sampler,
            verbose=False,
        )
        elapsed = time.perf_counter() - generation_started
        final_tokens = list(state["tokenizer"].encode(raw, add_special_tokens=False))
        parsed = parse_semantic_proposal(raw, config)
        controlled = controller_output(parsed, config)
        persist_access()
        return {
            "name": record["record_id"],
            "record_id": record["record_id"],
            "display_position": record["display_position"],
            "structurally_valid": parsed["structurally_valid"],
            "invalid_reason": parsed.get("invalid_reason"),
            "semantic_proposal": (
                {key: value for key, value in parsed.items() if key not in {"structurally_valid", "invalid_reason"}}
                if parsed["structurally_valid"]
                else None
            ),
            "controller_output": controlled,
            "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
            "bounded_final_prompt_sha256": sha256_json(final_prompt_tokens),
            "reasoning_response_sha256": hashlib.sha256(reasoning_text.encode("utf-8")).hexdigest(),
            "raw_final_response_sha256": hashlib.sha256(raw.encode("utf-8")).hexdigest(),
            "prompt_token_count": len(prompt_tokens),
            "reasoning_phase_generated_token_count": len(reasoning_tokens),
            "reasoning_phase_retained_token_count": retained_count,
            "reasoning_naturally_closed_within_budget": natural_close,
            "reasoning_phase_maximum_tokens_hit": len(reasoning_tokens)
            >= config["model"]["reasoning_phase_maximum_tokens"],
            "final_phase_generated_token_count": len(final_tokens),
            "final_phase_maximum_tokens_hit": len(final_tokens)
            >= config["model"]["final_phase_maximum_tokens"],
            "generation_seconds": elapsed,
            "raw_final_response_persisted": False,
            "reasoning_response_persisted": False,
        }

    summary_holder: dict[str, Any] = {}

    def evaluate_gates(completed: dict[str, dict[str, Any]]) -> dict[str, bool]:
        rows = list(completed.values())
        structural_rate = sum(row["structurally_valid"] for row in rows) / len(rows)
        truncation_rate = sum(row["final_phase_maximum_tokens_hit"] for row in rows) / len(rows)
        max_questions = max(
            (len(row["controller_output"]["clarification_questions"]) for row in rows), default=0
        )
        routes = {route: sum(row["controller_output"]["route"] == route for row in rows) for route in ("PLAN", "CLARIFY", "DEFER")}
        summary_holder.update(
            {
                "record_count": len(rows),
                "structurally_valid_count": sum(row["structurally_valid"] for row in rows),
                "structural_validity_rate": structural_rate,
                "safe_fallback_count": sum(row["controller_output"]["used_safe_fallback"] for row in rows),
                "final_phase_token_limit_hit_count": sum(row["final_phase_maximum_tokens_hit"] for row in rows),
                "final_phase_token_limit_hit_rate": truncation_rate,
                "maximum_questions_per_clarification": max_questions,
                "route_counts": routes,
            }
        )
        gates = config["qualification_gates"]
        return {
            "required_record_count": len(rows) == gates["required_public_request_count"],
            "minimum_structural_validity_rate": structural_rate >= gates["minimum_structural_validity_rate"],
            "maximum_final_phase_token_limit_hit_rate": truncation_rate <= gates["maximum_final_phase_token_limit_hit_rate"],
            "required_controller_coverage_rate": len(rows) / gates["required_public_request_count"] >= gates["required_controller_coverage_rate"],
            "maximum_questions_per_clarification": max_questions <= gates["maximum_questions_per_clarification"],
            "maximum_retry_count": access["retry_count"] <= gates["maximum_retry_count"],
            "maximum_api_call_count": access["api_call_count"] <= gates["maximum_api_call_count"],
            "maximum_real_service_call_count": access["real_service_call_count"] <= gates["maximum_real_service_call_count"],
            "maximum_external_side_effect_count": access["external_side_effect_count"] <= gates["maximum_external_side_effect_count"],
            "maximum_actual_execution_count": access["actual_execution_count"] <= gates["maximum_actual_execution_count"],
        }

    census = run_locked_census_once(
        output_dir=OUTPUT_DIR / "census",
        attempt={
            "condition": config["experiment"],
            "model_repository": config["model"]["repository"],
            "model_revision": config["model"]["revision"],
            "input_sha256": config["participant"]["public_requests_sha256"],
            "reasoning_phase_maximum_tokens": config["model"]["reasoning_phase_maximum_tokens"],
            "final_phase_maximum_tokens": config["model"]["final_phase_maximum_tokens"],
        },
        fixture_rows=[{"name": row["record_id"], "record": row} for row in records],
        evaluate_fixture=evaluate_record,
        evaluate_gates=evaluate_gates,
        result_metadata={
            "schema_version": "prospective-language-pilot-v1-phase2-census-result",
            "claim_boundary": config["claim_boundary"],
        },
        pass_decision=config["post_run_rule"]["if_all_gates_pass"],
        fail_decision=config["post_run_rule"]["otherwise"],
    )
    persist_access()

    ordered = [census["fixtures"][record["record_id"]] for record in records]
    participant_outputs: list[dict[str, Any]] = []
    private_outputs: list[dict[str, Any]] = []
    for source, row in zip(records, ordered):
        controlled = row["controller_output"]
        participant_payload = {
            "schema_version": "1.0.0",
            "study_id": source["study_id"],
            "participant_code": source["participant_code"],
            "record_id": source["record_id"],
            "display_position": source["display_position"],
            "route": controlled["route"],
            "clarification_questions": controlled["clarification_questions"],
            "sandbox_plan": controlled["sandbox_plan"],
            "defer_message": controlled["defer_message"],
            "used_safe_fallback": controlled["used_safe_fallback"],
        }
        participant_outputs.append(
            {**participant_payload, "controller_payload_sha256": sha256_json(participant_payload)}
        )
        private_outputs.append(
            {
                **participant_payload,
                "semantic_proposal": row["semantic_proposal"],
                "structurally_valid": row["structurally_valid"],
                "invalid_reason": row["invalid_reason"],
                "prompt_sha256": row["prompt_sha256"],
                "reasoning_response_sha256": row["reasoning_response_sha256"],
                "raw_final_response_sha256": row["raw_final_response_sha256"],
            }
        )

    participant_bytes = jsonl_bytes(participant_outputs)
    private_bytes = jsonl_bytes(private_outputs)
    participant_path = OUTPUT_DIR / "participant" / "phase2_controller_outputs.jsonl"
    private_path = OUTPUT_DIR / "private" / "phase2_semantic_records.jsonl"
    write_bytes(participant_path, participant_bytes)
    write_bytes(private_path, private_bytes)
    result = {
        "schema_version": "prospective-language-pilot-v1-phase2-result",
        "completed": True,
        "qualified_for_clarification_batch": census["passed"],
        "decision": census["decision"],
        "gates": census["gates"],
        "summary": summary_holder,
        "access": access,
        "files": {
            "participant_controller_outputs": {
                "path": str(participant_path.relative_to(ROOT)),
                "sha256": sha256_bytes(participant_bytes),
                "record_count": len(participant_outputs),
            },
            "private_semantic_records": {
                "path": str(private_path.relative_to(ROOT)),
                "sha256": sha256_bytes(private_bytes),
                "record_count": len(private_outputs),
            },
        },
        "claim_boundary": config["claim_boundary"],
    }
    write_json(OUTPUT_DIR / "result.json", result)
    print(json.dumps({"qualified": census["passed"], **summary_holder, "elapsed_seconds": access["elapsed_seconds"]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
