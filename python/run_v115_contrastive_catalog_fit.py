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

from locked_census_harness import run_locked_census_once, write_json
from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT
from v100_massive_source import parse_massive_archive
from v104_massive_language_extraction import build_selected_language_artifacts
from v106_open_world_benchmark import (
    build_declared_training_records, character_retrieval_observations, fit_character_retrieval,
)
from v109_open_world_typed_choice import render_choice_prompt, validate_and_expand_choice
from v115_contrastive_catalog_fit import (
    classify_v115, evaluate_v115, render_contrastive_prompt, reviewed_choice,
    validate_and_expand_contrastive,
)


def payload_hash(value: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))


def extract_fresh_language(lock: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    archive_bytes = (PROJECT_ROOT / lock["source_archive"]).read_bytes()
    if hashlib.sha256(archive_bytes).hexdigest() != lock["source_archive_sha256"]:
        raise RuntimeError("V115 source archive mismatch")
    config = lock["config_payload"]
    source_records, member = parse_massive_archive(
        archive_bytes, config["extraction"]["expectedLocaleMemberSuffix"],
    )
    inventory = json.loads((PROJECT_ROOT / lock["source_inventory"]).read_text())
    population = json.loads((PROJECT_ROOT / lock["fresh_population"]).read_text())
    artifacts = build_selected_language_artifacts(
        population, inventory, source_records,
        {"canonicalSourcePartitionMap": config["extraction"]["canonicalSourcePartitionMap"]},
    )
    role = config["extraction"]["role"]
    records = artifacts["role_records"][role]
    expected = config["extraction"]
    class_counts: dict[str, int] = {}
    for row in records:
        class_counts[row["class_label"]] = class_counts.get(row["class_label"], 0) + 1
    gates = {
        "record_count": len(records) == expected["requiredRecordCount"],
        "balanced_class_counts": all(
            class_counts.get(label) == expected["requiredRecordCountPerClass"]
            for label in config["freshPopulation"]["classes"]
        ),
        "exact_selected_identifier_set": artifacts["exact_selected_identifier_set"],
        "exact_structural_ground_truth_match": artifacts["exact_structural_ground_truth_match"],
        "exact_familiarity_reconstruction": artifacts["exact_familiarity_reconstruction"],
        "exact_slot_type_count_reconstruction": artifacts["exact_slot_type_count_reconstruction"],
        "zero_unselected_language": artifacts["unselected_language_record_count"] <= expected["maximumUnselectedLanguageRecordCount"],
    }
    if not all(gates.values()):
        raise RuntimeError(f"V115 fresh extraction gate failure: {gates}")
    return records, {"source_locale_member": member, "class_counts": class_counts, "gates": gates}


def main() -> None:
    lock_path = PROJECT_ROOT / "configs/v115-contrastive-catalog-fit-lock.json"
    language_path = PROJECT_ROOT / "outputs/v115-contrastive-catalog-fit/fresh-language/development-contrastive.jsonl"
    output_dir = PROJECT_ROOT / "outputs/v115-contrastive-catalog-fit/model-contrastive"
    if language_path.exists() or output_dir.exists():
        raise RuntimeError("V115 extraction and model condition may run only once")
    lock = json.loads(lock_path.read_text())
    if payload_hash({key: value for key, value in lock.items() if key != "lock_payload_sha256"}) != lock["lock_payload_sha256"]:
        raise RuntimeError("V115 lock mismatch")
    dependency_keys = (
        "config", "parent_outcome", "parent_analysis_lock", "V112_lock", "source_inventory",
        "source_archive", "V101_population", "V112_population", "V114_population",
        "visible_catalog", "choice_catalog", "model_manifest", "baseline_lock", "V109_result",
        "fresh_population", "plan", "protocol", "tests", "runner", "verifier", "auditor",
        "design_audit",
    )
    for key in dependency_keys:
        if file_sha256(PROJECT_ROOT / lock[key]) != lock[f"{key}_sha256"]:
            raise RuntimeError(f"V115 dependency drifted: {key}")

    config = lock["config_payload"]
    v112_config = lock["V112_config_payload"]
    records, extraction_summary = extract_fresh_language(lock)
    write_jsonl(language_path, records)
    archive_bytes = (PROJECT_ROOT / lock["source_archive"]).read_bytes()
    source_records, _ = parse_massive_archive(
        archive_bytes, config["extraction"]["expectedLocaleMemberSuffix"],
    )
    visible_catalog = json.loads((PROJECT_ROOT / lock["visible_catalog"]).read_text())
    training = build_declared_training_records(source_records, visible_catalog)
    retrieval_spec = lock["baseline_config_payload"]["deterministicBaselines"]["character_ngram_retrieval"]
    fitted = fit_character_retrieval(training, retrieval_spec["vectorizer"])
    retrieval = character_retrieval_observations(fitted, records)
    choice_catalog = json.loads((PROJECT_ROOT / lock["choice_catalog"]).read_text())
    manifest = json.loads((PROJECT_ROOT / lock["model_manifest"]).read_text())
    snapshot = Path(manifest["snapshot_path"])
    fixtures = [
        {"name": row["record_id"], "kind": "observed_fresh_contrastive", "record": row}
        for row in records
    ] + [
        {"name": f"v115::missing::{index:03d}", "kind": "controlled_missing_observation", "record": None}
        for index in range(config["condition"]["controlledMissingObservationCount"])
    ]
    access: dict[str, Any] = {
        "condition_id": config["condition"]["id"],
        "fresh_development_language_read_count": 1,
        "protected_test_language_read_count": 0,
        "manual_language_or_raw_response_inspection_count": 0,
        "model_load_count": 0, "model_generation_count": 0, "LLM_API_call_count": 0,
        "adapter_training_run_count": 0, "real_service_call_count": 0,
        "external_side_effect_count": 0,
    }
    state: dict[str, Any] = {"model": None, "tokenizer": None}
    mx.reset_peak_memory()
    started = time.perf_counter()

    def run_generation(system: str, payload: str, max_tokens: int, max_prompt_tokens: int) -> tuple[str, int, int, float]:
        tokenizer = state["tokenizer"]
        prompt = tokenizer.apply_chat_template(
            [{"role": "system", "content": system}, {"role": "user", "content": payload}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False,
        )
        prompt_tokens = tokenizer.encode(prompt)
        if len(prompt_tokens) > max_prompt_tokens:
            raise RuntimeError("V115 prompt exceeds frozen token budget")
        generation_started = time.perf_counter()
        raw = generate(
            state["model"], tokenizer, prompt=prompt, max_tokens=max_tokens,
            sampler=make_sampler(temp=0.0), verbose=False,
        )
        access["model_generation_count"] += 1
        access["elapsed_seconds"] = time.perf_counter() - started
        access["peak_active_memory_bytes"] = int(mx.get_peak_memory())
        write_json(output_dir / "access-progress.json", access)
        return raw, len(prompt_tokens), len(tokenizer.encode(raw)), time.perf_counter() - generation_started

    def evaluate_fixture(fixture: dict[str, Any]) -> dict[str, Any]:
        if state["model"] is None:
            load_started = time.perf_counter()
            model, tokenizer = load(str(snapshot))
            model.eval()
            state["model"], state["tokenizer"] = model, tokenizer
            access["model_load_count"] += 1
            access["model_load_seconds"] = time.perf_counter() - load_started
        observed = fixture["kind"] == "observed_fresh_contrastive"
        utterance = fixture["record"]["utterance"] if observed else None

        first_payload = render_choice_prompt(choice_catalog, utterance, observed, v112_config)
        first_raw, first_prompt_tokens, first_generated_tokens, first_seconds = run_generation(
            v112_config["prompt"]["system"], first_payload,
            v112_config["decoding"]["maximumNewTokens"],
            v112_config["prompt"]["maximumPromptTokens"],
        )
        first_parsed, first_valid, first_reason = validate_and_expand_choice(
            first_raw, choice_catalog, v112_config,
        )
        nearest_intent = retrieval[fixture["name"]]["nearest_intent"] if observed else None
        candidate = reviewed_choice(first_parsed, nearest_intent, choice_catalog, observed)
        write_json(output_dir / "stage-progress.json", {
            "active_fixture": fixture["name"], "completed_stage": "pass_one",
            "pass_one_raw_response_sha256": hashlib.sha256(first_raw.encode()).hexdigest(),
            "pass_one_parsed_response": first_parsed,
            "pass_one_response_valid": first_valid,
            "model_generation_count": access["model_generation_count"],
        })

        second_payload = render_contrastive_prompt(choice_catalog, candidate, utterance, observed, config)
        second_raw, second_prompt_tokens, second_generated_tokens, second_seconds = run_generation(
            config["prompt"]["system"], second_payload,
            config["decoding"]["maximumNewTokens"], config["prompt"]["maximumPromptTokens"],
        )
        second_parsed, second_evidence, second_valid, second_reason = validate_and_expand_contrastive(
            second_raw, candidate, choice_catalog, config,
        )
        return {
            "name": fixture["name"], "kind": fixture["kind"],
            "candidate_choice_id": candidate["choice_id"],
            "pass_one": {
                "raw_response": first_raw,
                "raw_response_sha256": hashlib.sha256(first_raw.encode()).hexdigest(),
                "parsed_response": first_parsed, "response_valid": first_valid,
                "validation_reason": first_reason, "prompt_token_count": first_prompt_tokens,
                "generated_token_count": first_generated_tokens, "generation_seconds": first_seconds,
            },
            "pass_two": {
                "raw_response": second_raw,
                "raw_response_sha256": hashlib.sha256(second_raw.encode()).hexdigest(),
                "parsed_response": second_parsed, "evidence": second_evidence,
                "response_valid": second_valid, "validation_reason": second_reason,
                "prompt_token_count": second_prompt_tokens,
                "generated_token_count": second_generated_tokens, "generation_seconds": second_seconds,
            },
            "permanently_non_authoritative": True,
            "safe_hypothesis_universe_pruned": False,
            "capability_defined": False, "executable": False,
        }

    def evaluate_gates(completed: dict[str, dict[str, Any]]) -> dict[str, bool]:
        summary = evaluate_v115(
            records, completed, retrieval, access, v112_config, config,
            lock["baseline_config_payload"],
        )
        evaluate_gates.last_summary = summary
        return {
            **{f"contrastive_evidence::{key}": value for key, value in summary["contrastive_evidence_gates"].items()},
            **{f"combined_policy::{key}": value for key, value in summary["combined_quality_gates"].items()},
            **{f"access::{key}": value for key, value in summary["access_gates"].items()},
        }

    evaluate_gates.last_summary = None
    result = run_locked_census_once(
        output_dir=output_dir, attempt=access, fixture_rows=fixtures,
        evaluate_fixture=evaluate_fixture, evaluate_gates=evaluate_gates,
        result_metadata={
            "schema_version": "115-contrastive-catalog-fit-result",
            "experiment": config["experiment"], "condition": config["condition"],
            "model_condition": v112_config["condition"],
            "fresh_population_sha256": json.loads((PROJECT_ROOT / lock["fresh_population"]).read_text())["selected_population_sha256"],
            "fresh_language_sha256": file_sha256(language_path),
            "extraction_summary": extraction_summary,
            "claim_boundary": "record-disjoint controlled open-set MASSIVE development evidence from the same source distribution; two local non-authoritative semantic passes; no protected language, API, training, induction, capability or action authority, execution, service call, or side effect",
        },
        pass_decision="all_contrastive_evidence_combined_policy_and_access_gates_pass",
        fail_decision="one_or_more_contrastive_evidence_combined_policy_or_access_gates_fail",
    )
    summary = evaluate_gates.last_summary
    flags = classify_v115(summary)
    access["elapsed_seconds"] = time.perf_counter() - started
    access["peak_active_memory_bytes"] = int(mx.get_peak_memory())
    completed = bool(
        len(result["fixtures"]) == config["condition"]["observedFixtureCount"] + config["condition"]["controlledMissingObservationCount"]
        and access["model_generation_count"] == config["condition"]["totalGenerationCount"]
    )
    result.update({
        "summary": summary, **flags, "completed_condition": completed,
        "decision": flags["decision"], "final_access": access,
        "output_integrity": {
            "fresh_language": {"path": str(language_path.relative_to(PROJECT_ROOT)), "sha256": file_sha256(language_path)},
        },
    })
    write_json(output_dir / "result.json", result)
    write_json(output_dir / "access.json", access)
    print(json.dumps({
        "completed_condition": completed, **flags, "summary": summary, "access": access,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
