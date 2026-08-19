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
from v144_local_certificate_realization import evaluate, parse_certificate_response, render_prompt


def main() -> None:
    lock_path = PROJECT_ROOT / "configs/v144-local-certificate-realization-lock.json"
    output_dir = PROJECT_ROOT / "outputs/v144-local-certificate-realization/model-realization"
    if output_dir.exists():
        raise RuntimeError("V144 exact development realization may run only once")
    lock = json.loads(lock_path.read_text())
    if payload_hash({key: value for key, value in lock.items() if key != "lock_payload_sha256"}) != lock["lock_payload_sha256"]:
        raise RuntimeError("V144 lock mismatch")
    dependencies = [key for key in lock if not key.endswith("_sha256") and f"{key}_sha256" in lock]
    for key in dependencies:
        if file_sha256(PROJECT_ROOT / lock[key]) != lock[f"{key}_sha256"]:
            raise RuntimeError(f"V144 dependency drifted: {key}")
    if not lock["authorization"]["run_exact_single_pinned_local_development_realization"]:
        raise RuntimeError("V144 model run not authorized")
    config = lock["config_payload"]
    public_rows = json.loads((PROJECT_ROOT / lock["development_public_fixtures"]).read_text())
    hidden_rows = json.loads((PROJECT_ROOT / lock["development_hidden_fixtures"]).read_text())
    catalog = json.loads((PROJECT_ROOT / lock["choice_catalog"]).read_text())
    v142_config = json.loads((PROJECT_ROOT / lock["V142_config"]).read_text())
    v136_config = json.loads((PROJECT_ROOT / lock["V136_config"]).read_text())
    manifest = json.loads((PROJECT_ROOT / lock["model_manifest"]).read_text())
    snapshot = Path(manifest["snapshot_path"])
    if len(public_rows) != config["population"]["fixtureCount"] or any(row["split"] != "development" for row in public_rows):
        raise RuntimeError("V144 locked development population mismatch")
    fixture_rows = [{"name": row["fixture_id"], "fixture": row} for row in public_rows]
    access: dict[str, Any] = {
        "V134_language_read_count": 0,
        "external_language_read_count": 0,
        "model_load_count": 0,
        "model_generation_count": 0,
        "maximum_generation_count_per_fixture": 0,
        "test_fixture_model_generation_count": 0,
        "retry_count": 0,
        "manual_raw_response_or_trace_inspection_count": 0,
        "persisted_raw_response_or_trace_count": 0,
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
            access["model_load_seconds"] = time.perf_counter() - load_started
        fixture = row["fixture"]
        payload = render_prompt(catalog, fixture, config)
        prompt = state["tokenizer"].apply_chat_template(
            [{"role": "system", "content": config["prompt"]["system"]}, {"role": "user", "content": payload}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=config["model"]["enableThinking"],
        )
        prompt_tokens = state["tokenizer"].encode(prompt)
        if len(prompt_tokens) > config["prompt"]["maximumPromptTokens"]:
            raise RuntimeError("V144 prompt exceeds frozen budget")
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
        access["model_generation_count"] += 1
        access["maximum_generation_count_per_fixture"] = 1
        access["elapsed_seconds"] = time.perf_counter() - started
        access["peak_active_memory_bytes"] = int(mx.get_peak_memory())
        write_json(output_dir / "access-progress.json", access)
        parsed = parse_certificate_response(raw, catalog, v142_config)
        final_text = parsed.pop("final_text")
        generated_tokens = state["tokenizer"].encode(raw)
        final_tokens = state["tokenizer"].encode(final_text) if final_text else []
        return {
            "name": row["name"],
            "fixture_id": fixture["fixture_id"],
            **parsed,
            "raw_response_sha256": hashlib.sha256(raw.encode("utf-8")).hexdigest(),
            "final_response_sha256": hashlib.sha256(final_text.encode("utf-8")).hexdigest(),
            "prompt_token_count": len(prompt_tokens),
            "generated_token_count": len(generated_tokens),
            "final_token_count": len(final_tokens),
            "reasoning_or_prefix_token_count": max(0, len(generated_tokens) - len(final_tokens)),
            "maximum_new_tokens_hit": len(generated_tokens) >= config["model"]["maximumNewTokens"],
            "generation_seconds": generation_seconds,
            "raw_response_or_trace_persisted": False,
            "permanently_non_authoritative": True,
            "authoritative_hypothesis_universe_pruned": False,
            "capability_defined": False,
            "executable": False,
            "actual_execution_count": 0,
        }

    def aggregate(completed: dict[str, dict[str, Any]]) -> dict[str, bool]:
        summary = evaluate(completed, hidden_rows, catalog, v136_config, access, config)
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
            "schema_version": "144-local-certificate-realization-result",
            "experiment": config["experiment"],
            "development_fixture_count": len(public_rows),
            "test_fixture_model_generation_count": 0,
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
