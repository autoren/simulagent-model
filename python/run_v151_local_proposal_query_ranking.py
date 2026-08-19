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

from audit_and_freeze_v122_prequery_signal_inventory import payload_hash
from locked_census_harness import run_locked_census_once, write_json
from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT
from v151_local_proposal_query_ranking import evaluate, parse_proposal, render_prompt


def main() -> None:
    lock_path = PROJECT_ROOT / "configs/v151-local-proposal-query-ranking-lock.json"
    output_dir = PROJECT_ROOT / "outputs/v151-local-proposal-query-ranking/model-realization"
    if output_dir.exists():
        raise RuntimeError("V151 exact development realization may run only once")
    lock = json.loads(lock_path.read_text())
    if payload_hash({key: value for key, value in lock.items() if key != "lock_payload_sha256"}) != lock["lock_payload_sha256"]:
        raise RuntimeError("V151 lock mismatch")
    dependencies = [key for key in lock if not key.endswith("_sha256") and f"{key}_sha256" in lock]
    for key in dependencies:
        if file_sha256(PROJECT_ROOT / lock[key]) != lock[f"{key}_sha256"]:
            raise RuntimeError(f"V151 dependency drifted: {key}")
    authorization = lock["authorization"]
    if (
        not authorization["run_exact_single_pinned_local_development_realization"]
        or authorization["generate_on_closed_answer_or_V149_evaluation_fixtures"]
        or authorization["persist_or_manually_inspect_raw_model_responses"]
        or authorization["run_API_training_induction_authority_action_or_execution"]
    ):
        raise RuntimeError("V151 model run authorization is invalid")

    config = lock["config_payload"]
    public_rows = json.loads((PROJECT_ROOT / lock["development_public_fixtures"]).read_text())
    hidden_rows = json.loads((PROJECT_ROOT / lock["development_hidden_fixtures"]).read_text())
    answer_metadata = json.loads((PROJECT_ROOT / lock["development_answer_metadata"]).read_text())
    catalog = json.loads((PROJECT_ROOT / lock["interaction_catalog"]).read_text())
    witness_config = json.loads((PROJECT_ROOT / lock["witness_config"]).read_text())
    oracle_config = json.loads((PROJECT_ROOT / lock["oracle_config"]).read_text())
    manifest = json.loads((PROJECT_ROOT / lock["model_manifest"]).read_text())
    snapshot = Path(manifest["snapshot_path"])
    if (
        len(public_rows) != config["population"]["requestFixtureCount"]
        or any(row["split"] != "development" or row["closed_answer_event"] is not None for row in public_rows)
    ):
        raise RuntimeError("V151 locked development request population mismatch")

    fixture_rows = [{"name": row["fixture_id"], "fixture": row} for row in public_rows]
    access: dict[str, Any] = {
        "tokenizer_load_count": 0,
        "model_load_count": 0,
        "model_generation_count": 0,
        "maximum_generation_count_per_fixture": 0,
        "closed_answer_model_generation_count": 0,
        "evaluation_fixture_model_generation_count": 0,
        "retry_count": 0,
        "manual_raw_response_inspection_count": 0,
        "persisted_raw_response_count": 0,
        "API_call_count": 0,
        "training_run_count": 0,
        "real_service_call_count": 0,
        "external_side_effect_count": 0,
        "actual_execution_count": 0,
    }
    state: dict[str, Any] = {"model": None, "tokenizer": None}
    mx.reset_peak_memory()
    started = time.perf_counter()

    def evaluate_fixture(row: dict[str, Any]) -> dict[str, Any]:
        if state["model"] is None:
            load_started = time.perf_counter()
            model, tokenizer = load(str(snapshot))
            model.eval()
            state["model"], state["tokenizer"] = model, tokenizer
            access["model_load_count"] += 1
            access["tokenizer_load_count"] += 1
            access["model_load_seconds"] = time.perf_counter() - load_started
        fixture = row["fixture"]
        payload = render_prompt(catalog, fixture, config)
        prompt = state["tokenizer"].apply_chat_template(
            [
                {"role": "system", "content": config["prompt"]["system"]},
                {"role": "user", "content": payload},
            ],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=config["model"]["enableThinking"],
        )
        prompt_tokens = state["tokenizer"].encode(prompt, add_special_tokens=False)
        if len(prompt_tokens) > config["prompt"]["maximumPromptTokens"]:
            raise RuntimeError("V151 prompt exceeds frozen budget")

        access["model_generation_count"] += 1
        access["maximum_generation_count_per_fixture"] = 1
        access["elapsed_seconds"] = time.perf_counter() - started
        access["peak_active_memory_bytes"] = int(mx.get_peak_memory())
        write_json(output_dir / "access-progress.json", access)
        generation_started = time.perf_counter()
        raw = generate(
            state["model"],
            state["tokenizer"],
            prompt=prompt,
            max_tokens=config["model"]["maximumNewTokens"],
            sampler=make_sampler(temp=config["model"]["temperature"]),
            verbose=False,
        )
        generation_seconds = time.perf_counter() - generation_started
        parsed = parse_proposal(raw, catalog, config)
        generated_tokens = state["tokenizer"].encode(raw, add_special_tokens=False)
        access["elapsed_seconds"] = time.perf_counter() - started
        access["peak_active_memory_bytes"] = int(mx.get_peak_memory())
        write_json(output_dir / "access-progress.json", access)
        return {
            "name": row["name"],
            "fixture_id": fixture["fixture_id"],
            **parsed,
            "raw_response_sha256": hashlib.sha256(raw.encode("utf-8")).hexdigest(),
            "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
            "prompt_token_count": len(prompt_tokens),
            "generated_token_count": len(generated_tokens),
            "maximum_new_tokens_hit": len(generated_tokens) >= config["model"]["maximumNewTokens"],
            "generation_seconds": generation_seconds,
            "raw_response_persisted": False,
        }

    def aggregate(completed: dict[str, dict[str, Any]]) -> dict[str, bool]:
        summary = evaluate(
            completed,
            hidden_rows,
            answer_metadata,
            catalog,
            witness_config,
            oracle_config,
            access,
            config,
        )
        aggregate.last_summary = summary
        return {
            **{f"qualification::{key}": value for key, value in summary["qualification_gates"].items()},
            **{f"access::{key}": value for key, value in summary["access_gates"].items()},
        }

    aggregate.last_summary = None
    result = run_locked_census_once(
        output_dir=output_dir,
        attempt=access,
        fixture_rows=fixture_rows,
        evaluate_fixture=evaluate_fixture,
        evaluate_gates=aggregate,
        result_metadata={
            "schema_version": "151-local-proposal-query-ranking-result",
            "experiment": config["experiment"],
            "development_request_fixture_count": len(public_rows),
            "closed_answer_model_generation_count": 0,
            "evaluation_fixture_model_generation_count": 0,
            "catalog_sha256": catalog["catalog_sha256"],
            "claim_boundary": config["claimBoundary"],
        },
        pass_decision=config["decisionRule"]["ifEveryQualificationAndAccessGatePasses"],
        fail_decision=config["decisionRule"]["otherwise"],
    )
    summary = aggregate.last_summary
    access["elapsed_seconds"] = time.perf_counter() - started
    access["peak_active_memory_bytes"] = int(mx.get_peak_memory())
    result.update(
        {
            "summary": summary,
            "decision": summary["decision"],
            "completed_condition": len(result["fixtures"]) == len(fixture_rows),
            "final_access": access,
        }
    )
    write_json(output_dir / "result.json", result)
    write_json(output_dir / "access.json", access)
    print(json.dumps({"completed_condition": result["completed_condition"], "decision": result["decision"], "summary": summary, "access": access}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
