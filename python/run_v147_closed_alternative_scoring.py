#!/usr/bin/env python3
from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path
import time
from typing import Any

import mlx.core as mx
import mlx.nn as nn
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache

from audit_and_freeze_v122_prequery_signal_inventory import payload_hash
from locked_census_harness import run_locked_census_once, write_json
from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT
from v147_closed_alternative_scoring import alias_mapping, evaluate, render_prompt, select_scored_code


def _score_aliases(
    model: Any,
    prompt_ids: list[int],
    alias_token_ids: dict[str, list[int]],
) -> dict[str, float]:
    prompt = mx.array(prompt_ids)[None]
    cache = make_prompt_cache(model)
    logits = None
    for start in range(0, prompt.shape[1], 2048):
        logits = model(prompt[:, start : start + 2048], cache=cache)
        mx.eval([entry.state for entry in cache])
        mx.clear_cache()
    if logits is None:
        raise RuntimeError("empty V147 prompt")
    prompt_logprobs = nn.log_softmax(logits[:, -1, :].astype(mx.float32))
    mx.eval(prompt_logprobs)

    scores: dict[str, float] = {}
    for alias, tokens in alias_token_ids.items():
        if not tokens:
            raise RuntimeError("empty V147 alias token sequence")
        score = float(prompt_logprobs[0, tokens[0]].item())
        if len(tokens) > 1:
            local_cache = copy.deepcopy(cache)
            continuation_inputs = mx.array(tokens[:-1])[None]
            continuation_targets = mx.array(tokens[1:])[None, :, None]
            continuation_logits = model(continuation_inputs, cache=local_cache)
            continuation_logprobs = nn.log_softmax(continuation_logits.astype(mx.float32))
            selected = mx.take_along_axis(continuation_logprobs, continuation_targets, axis=-1)[..., 0]
            mx.eval(selected)
            score += float(mx.sum(selected).item())
            del local_cache, continuation_inputs, continuation_targets, continuation_logits, continuation_logprobs, selected
            mx.clear_cache()
        if not math.isfinite(score):
            raise RuntimeError("non-finite V147 alias score")
        scores[alias] = score
    del prompt, cache, logits, prompt_logprobs
    mx.clear_cache()
    return scores


def main() -> None:
    lock_path = PROJECT_ROOT / "configs/v147-closed-alternative-scoring-lock.json"
    output_dir = PROJECT_ROOT / "outputs/v147-closed-alternative-scoring/model-scoring-realization"
    if output_dir.exists():
        raise RuntimeError("V147 exact development scoring realization may run only once")
    lock = json.loads(lock_path.read_text())
    if payload_hash({key: value for key, value in lock.items() if key != "lock_payload_sha256"}) != lock["lock_payload_sha256"]:
        raise RuntimeError("V147 lock mismatch")
    dependencies = [key for key in lock if not key.endswith("_sha256") and f"{key}_sha256" in lock]
    for key in dependencies:
        if file_sha256(PROJECT_ROOT / lock[key]) != lock[f"{key}_sha256"]:
            raise RuntimeError(f"V147 dependency drifted: {key}")
    if not lock["authorization"]["run_exact_single_pinned_local_development_scoring_realization"]:
        raise RuntimeError("V147 model scoring run not authorized")
    if lock["authorization"]["score_or_use_V146_test_split"]:
        raise RuntimeError("V147 lock cannot authorize retired V146 test use")

    config = lock["config_payload"]
    public_rows = json.loads((PROJECT_ROOT / lock["development_public_fixtures"]).read_text())
    hidden_rows = json.loads((PROJECT_ROOT / lock["development_hidden_fixtures"]).read_text())
    catalog = json.loads((PROJECT_ROOT / lock["choice_catalog"]).read_text())
    codebook = json.loads((PROJECT_ROOT / lock["certificate_codebook"]).read_text())["entries"]
    v136 = json.loads((PROJECT_ROOT / lock["V136_config"]).read_text())
    manifest = json.loads((PROJECT_ROOT / lock["model_manifest"]).read_text())
    snapshot = Path(manifest["snapshot_path"])
    if (
        len(public_rows) != config["population"]["fixtureCount"]
        or any(row["split"] != "development" for row in public_rows)
    ):
        raise RuntimeError("V147 locked development population mismatch")
    fixture_rows = [{"name": row["fixture_id"], "fixture": row} for row in public_rows]
    aliases = config["scoring"]["aliases"]
    access: dict[str, Any] = {
        "V134_language_read_count": 0,
        "external_language_read_count": 0,
        "tokenizer_load_count": 0,
        "model_load_count": 0,
        "model_generation_count": 0,
        "maximum_generation_count_per_fixture": 0,
        "test_fixture_model_generation_count": 0,
        "model_scoring_fixture_count": 0,
        "candidate_sequence_score_count": 0,
        "test_fixture_score_count": 0,
        "retry_count": 0,
        "manual_raw_response_or_trace_inspection_count": 0,
        "persisted_raw_response_or_trace_count": 0,
        "API_call_count": 0,
        "training_run_count": 0,
        "real_service_call_count": 0,
        "external_side_effect_count": 0,
        "actual_execution_count": 0,
    }
    state: dict[str, Any] = {"model": None, "tokenizer": None, "alias_token_ids": None}
    mx.reset_peak_memory()
    started = time.perf_counter()

    def evaluate_fixture(row: dict[str, Any]) -> dict[str, Any]:
        if state["model"] is None:
            load_started = time.perf_counter()
            model, tokenizer = load(str(snapshot))
            model.eval()
            alias_token_ids = {
                alias: tokenizer.encode(alias, add_special_tokens=False)
                for alias in aliases
            }
            if any(len(tokens) != config["scoring"]["requiredTokensPerAlias"] for tokens in alias_token_ids.values()):
                raise RuntimeError("V147 runtime alias tokenization drift")
            state["model"], state["tokenizer"], state["alias_token_ids"] = model, tokenizer, alias_token_ids
            access["model_load_count"] += 1
            access["tokenizer_load_count"] += 1
            access["model_load_seconds"] = time.perf_counter() - load_started

        fixture = row["fixture"]
        payload = render_prompt(catalog, codebook, fixture, config)
        prompt = state["tokenizer"].apply_chat_template(
            [
                {"role": "system", "content": config["prompt"]["system"]},
                {"role": "user", "content": payload},
            ],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=config["model"]["enableThinking"],
        )
        prompt_ids = state["tokenizer"].encode(prompt, add_special_tokens=False)
        if len(prompt_ids) > config["prompt"]["maximumPromptTokens"]:
            raise RuntimeError("V147 prompt exceeds frozen budget")
        for alias, tokens in state["alias_token_ids"].items():
            full_ids = state["tokenizer"].encode(prompt + alias, add_special_tokens=False)
            if full_ids[: len(prompt_ids)] != prompt_ids or full_ids[len(prompt_ids) :] != tokens:
                raise RuntimeError("V147 runtime prompt-alias token boundary drift")

        scoring_started = time.perf_counter()
        scores = _score_aliases(state["model"], prompt_ids, state["alias_token_ids"])
        scoring_seconds = time.perf_counter() - scoring_started
        selected = select_scored_code(fixture["fixture_id"], scores, codebook, config)
        mapping = alias_mapping(fixture["fixture_id"], codebook, aliases)
        access["model_scoring_fixture_count"] += 1
        access["candidate_sequence_score_count"] += len(scores)
        access["elapsed_seconds"] = time.perf_counter() - started
        access["peak_active_memory_bytes"] = int(mx.get_peak_memory())
        write_json(output_dir / "access-progress.json", access)
        return {
            "name": row["name"],
            "fixture_id": fixture["fixture_id"],
            **selected,
            "scores_by_alias": scores,
            "alias_mapping_sha256": hashlib.sha256(
                json.dumps(mapping, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest(),
            "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
            "prompt_token_count": len(prompt_ids),
            "candidate_sequence_score_count": len(scores),
            "scoring_seconds": scoring_seconds,
            "raw_response_or_trace_persisted": False,
            "permanently_non_authoritative": True,
            "authoritative_hypothesis_universe_pruned": False,
            "capability_defined": False,
            "executable": False,
            "actual_execution_count": 0,
        }

    def aggregate(completed: dict[str, dict[str, Any]]) -> dict[str, bool]:
        summary = evaluate(completed, hidden_rows, catalog, v136, access, config)
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
            "schema_version": "147-closed-alternative-scoring-result",
            "experiment": config["experiment"],
            "development_fixture_count": len(public_rows),
            "V146_test_fixture_score_count": 0,
            "V146_test_split_retired": True,
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
