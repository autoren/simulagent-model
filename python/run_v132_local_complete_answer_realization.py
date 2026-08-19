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
from v132_local_complete_answer_realization import (
    evaluate_realization, extract_selected_language_and_definitions, render_prompt, validate_answer,
)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))


def main() -> None:
    lock_path = PROJECT_ROOT / "configs/v132-local-complete-answer-realization-lock.json"
    output_dir = PROJECT_ROOT / "outputs/v132-local-complete-answer-realization/model-realization"
    language_path = PROJECT_ROOT / "outputs/v132-local-complete-answer-realization/selected-language/records.jsonl"
    prompt_catalog_path = PROJECT_ROOT / "outputs/v132-local-complete-answer-realization/selected-language/prompt-choice-catalog.json"
    if output_dir.exists() or language_path.exists() or prompt_catalog_path.exists():
        raise RuntimeError("V132 exact condition may run only once")
    lock = json.loads(lock_path.read_text())
    if payload_hash({key: value for key, value in lock.items() if key != "lock_payload_sha256"}) != lock["lock_payload_sha256"]:
        raise RuntimeError("V132 lock mismatch")
    dependencies = [key for key in lock if not key.endswith("_sha256") and f"{key}_sha256" in lock]
    for key in dependencies:
        if file_sha256(PROJECT_ROOT / lock[key]) != lock[f"{key}_sha256"]:
            raise RuntimeError(f"V132 dependency drifted: {key}")
    config = lock["config_payload"]
    population = json.loads((PROJECT_ROOT / lock["fixture_population"]).read_text())
    catalog = json.loads((PROJECT_ROOT / lock["choice_catalog"]).read_text())
    baseline = json.loads((PROJECT_ROOT / lock["baseline_config"]).read_text())
    v130 = json.loads((PROJECT_ROOT / lock["V130_config"]).read_text())
    archive_bytes = (PROJECT_ROOT / lock["source_archive"]).read_bytes()
    language, prompt_choices, extraction_summary = extract_selected_language_and_definitions(archive_bytes, population, catalog, config)
    write_jsonl(language_path, language); write_json(prompt_catalog_path, {"choices": prompt_choices})
    language_by_fixture = {row["fixture_id"]: row["utterance"] for row in language}
    fixture_rows = [
        {"name": row["fixture_id"], "structural": row, "utterance": language_by_fixture.get(row["fixture_id"])}
        for row in population["fixtures"]
    ]
    manifest = json.loads((PROJECT_ROOT / lock["model_manifest"]).read_text())
    snapshot = Path(manifest["snapshot_path"])
    access: dict[str, Any] = {
        "condition_id": config["condition"]["id"], "source_archive_read_count": 1,
        "automatic_selected_language_parse_count": 1, "persisted_selected_language_record_count": len(language),
        "manual_language_or_raw_response_inspection_count": 0, "original_protected_language_read_count": 0,
        "model_load_count": 0, "model_generation_count": 0, "LLM_API_call_count": 0,
        "adapter_training_run_count": 0, "real_service_call_count": 0, "external_side_effect_count": 0,
    }
    state: dict[str, Any] = {"model": None, "tokenizer": None}; mx.reset_peak_memory(); started = time.perf_counter()

    def evaluate_fixture(fixture: dict[str, Any]) -> dict[str, Any]:
        if state["model"] is None:
            load_started = time.perf_counter(); model, tokenizer = load(str(snapshot)); model.eval()
            state["model"], state["tokenizer"] = model, tokenizer
            access["model_load_count"] += 1; access["model_load_seconds"] = time.perf_counter() - load_started
        structural = fixture["structural"]; observed = structural["observation_available"]
        payload = render_prompt(prompt_choices, fixture["utterance"], observed, structural["presented_candidate_choice_id"], config)
        prompt = state["tokenizer"].apply_chat_template(
            [{"role": "system", "content": config["prompt"]["system"]}, {"role": "user", "content": payload}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False,
        )
        prompt_tokens = state["tokenizer"].encode(prompt)
        if len(prompt_tokens) > config["prompt"]["maximumPromptTokens"]: raise RuntimeError("V132 prompt exceeds frozen budget")
        generation_started = time.perf_counter()
        raw = generate(
            state["model"], state["tokenizer"], prompt=prompt,
            max_tokens=config["decoding"]["maximumNewTokens"], sampler=make_sampler(temp=0.0), verbose=False,
        )
        seconds = time.perf_counter() - generation_started; access["model_generation_count"] += 1
        access["elapsed_seconds"] = time.perf_counter() - started; access["peak_active_memory_bytes"] = int(mx.get_peak_memory())
        write_json(output_dir / "access-progress.json", access)
        answer, valid, reason = validate_answer(raw, catalog)
        return {
            "name": fixture["name"], "raw_response": raw,
            "raw_response_sha256": hashlib.sha256(raw.encode()).hexdigest(),
            "answer_choice_id": answer, "response_valid": valid, "validation_reason": reason,
            "prompt_token_count": len(prompt_tokens), "generated_token_count": len(state["tokenizer"].encode(raw)),
            "generation_seconds": seconds, "permanently_non_authoritative": True,
            "safe_hypothesis_universe_pruned": False, "capability_defined": False, "executable": False,
        }

    def evaluate_gates(completed: dict[str, dict[str, Any]]) -> dict[str, bool]:
        summary = evaluate_realization(population, completed, catalog, baseline, v130, access, config)
        evaluate_gates.last_summary = summary
        return {
            **{f"evidence::{key}": value for key, value in summary["evidence_gates"].items()},
            **{f"downstream::{key}": value for key, value in summary["downstream_gates"].items()},
            **{f"access::{key}": value for key, value in summary["access_gates"].items()},
        }

    evaluate_gates.last_summary = None
    result = run_locked_census_once(
        output_dir=output_dir, attempt=access, fixture_rows=fixture_rows,
        evaluate_fixture=evaluate_fixture, evaluate_gates=evaluate_gates,
        result_metadata={
            "schema_version": "132-local-complete-answer-realization-result", "experiment": config["experiment"],
            "condition": config["condition"], "extraction_summary": extraction_summary,
            "population_sha256": population["fixtures_sha256"], "catalog_sha256": catalog["catalog_sha256"],
            "selected_language_sha256": file_sha256(language_path), "prompt_choice_catalog_sha256": file_sha256(prompt_catalog_path),
            "claim_boundary": config["claimBoundary"],
        },
        pass_decision=config["decisionRule"]["ifEvidenceDownstreamAndAccessGatesPass"],
        fail_decision=config["decisionRule"]["otherwise"],
    )
    summary = evaluate_gates.last_summary
    access["elapsed_seconds"] = time.perf_counter() - started; access["peak_active_memory_bytes"] = int(mx.get_peak_memory())
    result.update({
        "summary": summary, "decision": summary["decision"], "completed_condition": len(result["fixtures"]) == config["condition"]["totalFixtureCount"],
        "final_access": access,
        "output_integrity": {
            "selected_language": {"path": str(language_path.relative_to(PROJECT_ROOT)), "sha256": file_sha256(language_path)},
            "prompt_choice_catalog": {"path": str(prompt_catalog_path.relative_to(PROJECT_ROOT)), "sha256": file_sha256(prompt_catalog_path)},
        },
    })
    write_json(output_dir / "result.json", result); write_json(output_dir / "access.json", access)
    print(json.dumps({"completed_condition": result["completed_condition"], "decision": result["decision"], "summary": summary, "access": access}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
