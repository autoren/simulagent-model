#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time
import traceback
from typing import Any

import mlx.core as mx
from mlx_lm import generate, load
from mlx_lm.sample_utils import make_sampler

from audit_and_freeze_v122_prequery_signal_inventory import payload_hash
from locked_census_harness import write_json
from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT
from v151_local_proposal_query_ranking import evaluate, parse_proposal, render_prompt
from v151r1_local_proposal_query_ranking_recovery import interrupted_fail_closed, recovery_evaluation_config


def main() -> None:
    lock_path = PROJECT_ROOT / "configs/v151r1-local-proposal-query-ranking-recovery-lock.json"
    output_dir = PROJECT_ROOT / "outputs/v151r1-local-proposal-query-ranking-recovery/model-recovery"
    if output_dir.exists():
        raise RuntimeError("V151r1 recovery may run only once")
    lock = json.loads(lock_path.read_text())
    if payload_hash({key: value for key, value in lock.items() if key != "lock_payload_sha256"}) != lock["lock_payload_sha256"]:
        raise RuntimeError("V151r1 recovery lock mismatch")
    dependencies = [key for key in lock if not key.endswith("_sha256") and f"{key}_sha256" in lock]
    for key in dependencies:
        if file_sha256(PROJECT_ROOT / lock[key]) != lock[f"{key}_sha256"]:
            raise RuntimeError(f"V151r1 dependency drifted: {key}")
    authorization = lock["authorization"]
    if (
        not authorization["retain_exact_58_hash_locked_outputs"]
        or not authorization["assign_interrupted_fixture_deterministic_invalid_fail_closed_output"]
        or not authorization["generate_exactly_37_never_started_fixtures_once"]
        or authorization["regenerate_persisted_or_interrupted_fixture"]
        or authorization["generate_on_closed_answer_or_V149_evaluation_fixtures"]
        or authorization["run_API_training_induction_authority_action_or_execution"]
    ):
        raise RuntimeError("V151r1 recovery authorization is invalid")

    recovery = lock["recovery_config_payload"]
    base = lock["base_V151_config_payload"]
    evaluation_config = recovery_evaluation_config(base, recovery)
    public = json.loads((PROJECT_ROOT / lock["development_public_fixtures"]).read_text())
    public_by_id = {row["fixture_id"]: row for row in public}
    hidden = json.loads((PROJECT_ROOT / lock["development_hidden_fixtures"]).read_text())
    answers = json.loads((PROJECT_ROOT / lock["development_answer_metadata"]).read_text())
    catalog = json.loads((PROJECT_ROOT / lock["interaction_catalog"]).read_text())
    witness = json.loads((PROJECT_ROOT / lock["witness_config"]).read_text())
    oracle = json.loads((PROJECT_ROOT / lock["oracle_config"]).read_text())
    manifest = json.loads((PROJECT_ROOT / lock["retained_partial_manifest"]).read_text())
    prior_access = json.loads((PROJECT_ROOT / lock["prior_access_progress"]).read_text())
    model_manifest = json.loads((PROJECT_ROOT / lock["model_manifest"]).read_text())
    snapshot = Path(model_manifest["snapshot_path"])

    completed: dict[str, dict[str, Any]] = {}
    for record in manifest["persisted_records"]:
        path = PROJECT_ROOT / record["path"]
        if file_sha256(path) != record["sha256"]:
            raise RuntimeError(f"retained V151 artifact drifted: {record['fixture_id']}")
        row = json.loads(path.read_text())
        if row.get("name") != record["fixture_id"] or row.get("fixture_id") != record["fixture_id"]:
            raise RuntimeError("retained V151 artifact identity mismatch")
        completed[record["fixture_id"]] = row
    interrupted_id = manifest["interrupted_fixture_id"]
    completed[interrupted_id] = interrupted_fail_closed(interrupted_id, catalog, base)

    access: dict[str, Any] = dict(prior_access)
    access["prior_interrupted_attempt_count"] = 1
    access["technical_fail_closed_fixture_count"] = 1
    access["recovery_model_load_count"] = 0
    access["recovery_model_generation_count"] = 0
    access["prior_elapsed_seconds"] = prior_access.get("elapsed_seconds", 0.0)
    access["recovery_elapsed_seconds"] = 0.0
    state: dict[str, Any] = {"model": None, "tokenizer": None}
    recovery_started = time.perf_counter()
    output_dir.mkdir(parents=True, exist_ok=False)
    write_json(output_dir / "attempt.json", access)
    mx.reset_peak_memory()

    try:
        for fixture_id in manifest["never_started_fixture_ids"]:
            fixture = public_by_id[fixture_id]
            if state["model"] is None:
                load_started = time.perf_counter()
                model, tokenizer = load(str(snapshot))
                model.eval()
                state["model"], state["tokenizer"] = model, tokenizer
                access["model_load_count"] += 1
                access["tokenizer_load_count"] += 1
                access["recovery_model_load_count"] += 1
                access["recovery_model_load_seconds"] = time.perf_counter() - load_started
            payload = render_prompt(catalog, fixture, base)
            prompt = state["tokenizer"].apply_chat_template(
                [
                    {"role": "system", "content": base["prompt"]["system"]},
                    {"role": "user", "content": payload},
                ],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=base["model"]["enableThinking"],
            )
            prompt_tokens = state["tokenizer"].encode(prompt, add_special_tokens=False)
            if len(prompt_tokens) > base["prompt"]["maximumPromptTokens"]:
                raise RuntimeError("V151r1 prompt exceeds frozen budget")
            access["model_generation_count"] += 1
            access["recovery_model_generation_count"] += 1
            access["maximum_generation_count_per_fixture"] = 1
            access["recovery_elapsed_seconds"] = time.perf_counter() - recovery_started
            access["elapsed_seconds"] = access["prior_elapsed_seconds"] + access["recovery_elapsed_seconds"]
            access["peak_active_memory_bytes"] = max(
                int(prior_access.get("peak_active_memory_bytes", 0)), int(mx.get_peak_memory())
            )
            write_json(output_dir / "access-progress.json", access)
            generation_started = time.perf_counter()
            raw = generate(
                state["model"],
                state["tokenizer"],
                prompt=prompt,
                max_tokens=base["model"]["maximumNewTokens"],
                sampler=make_sampler(temp=base["model"]["temperature"]),
                verbose=False,
            )
            generation_seconds = time.perf_counter() - generation_started
            parsed = parse_proposal(raw, catalog, base)
            generated_tokens = state["tokenizer"].encode(raw, add_special_tokens=False)
            row = {
                "name": fixture_id,
                "fixture_id": fixture_id,
                **parsed,
                "raw_response_sha256": hashlib.sha256(raw.encode("utf-8")).hexdigest(),
                "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
                "prompt_token_count": len(prompt_tokens),
                "generated_token_count": len(generated_tokens),
                "maximum_new_tokens_hit": len(generated_tokens) >= base["model"]["maximumNewTokens"],
                "generation_seconds": generation_seconds,
                "raw_response_persisted": False,
                "recovery_generated": True,
            }
            completed[fixture_id] = row
            ordinal = manifest["all_fixture_ids"].index(fixture_id)
            write_json(output_dir / "raw-fixtures" / f"{ordinal:03d}-{fixture_id}.json", row)
            access["recovery_elapsed_seconds"] = time.perf_counter() - recovery_started
            access["elapsed_seconds"] = access["prior_elapsed_seconds"] + access["recovery_elapsed_seconds"]
            access["peak_active_memory_bytes"] = max(
                int(prior_access.get("peak_active_memory_bytes", 0)), int(mx.get_peak_memory())
            )
            write_json(output_dir / "access-progress.json", access)

        if set(completed) != set(manifest["all_fixture_ids"]):
            raise RuntimeError("V151r1 combined completion mismatch")
        summary = evaluate(completed, hidden, answers, catalog, witness, oracle, access, evaluation_config)
        result = {
            "schema_version": "151r1-local-proposal-query-ranking-recovery-result",
            "experiment": recovery["experiment"],
            "completed_condition": True,
            "passed": summary["qualified"],
            "decision": summary["decision"],
            "retained_fixture_count": len(manifest["persisted_records"]),
            "technical_fail_closed_fixture_count": 1,
            "recovery_generated_fixture_count": len(manifest["never_started_fixture_ids"]),
            "closed_answer_model_generation_count": 0,
            "evaluation_fixture_model_generation_count": 0,
            "summary": summary,
            "fixtures": completed,
            "final_access": access,
            "claim_boundary": recovery["claimBoundary"],
        }
        write_json(output_dir / "result.json", result)
        write_json(output_dir / "access.json", access)
        print(json.dumps({"completed_condition": True, "decision": result["decision"], "summary": summary, "access": access}, indent=2, sort_keys=True))
    except Exception as error:
        failure = {
            "schema_version": "151r1-recovery-execution-failure",
            "status": "execution_failure",
            "completed_fixture_count": len(completed),
            "exception_type": type(error).__name__,
            "exception_message": str(error),
            "traceback": traceback.format_exc(),
            "access": access,
        }
        write_json(output_dir / "failure.json", failure)
        raise


if __name__ == "__main__":
    main()
