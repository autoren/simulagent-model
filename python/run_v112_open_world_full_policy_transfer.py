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
    ask_always_prediction, build_declared_training_records, character_retrieval_observations,
    evaluate_predictions, fit_character_retrieval, oracle_prediction, retrieval_prediction,
)
from v109_open_world_typed_choice import render_choice_prompt, validate_and_expand_choice
from v112_open_world_full_policy_transfer import (
    novelty_evidence_metrics, policy_prediction, policy_quality_gates,
)


def payload_hash(value: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))


def aggregate_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in metrics.items() if key != "scored_rows"}


def extract_fresh_language(lock: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    archive_bytes = (PROJECT_ROOT / lock["source_archive"]).read_bytes()
    if hashlib.sha256(archive_bytes).hexdigest() != lock["source_archive_sha256"]:
        raise RuntimeError("V112 source archive mismatch")
    source_records, member = parse_massive_archive(
        archive_bytes, lock["config_payload"]["extraction"]["expectedLocaleMemberSuffix"],
    )
    inventory = json.loads((PROJECT_ROOT / lock["source_inventory"]).read_text())
    population = json.loads((PROJECT_ROOT / lock["fresh_population"]).read_text())
    extraction_config = {
        "canonicalSourcePartitionMap": lock["config_payload"]["extraction"]["canonicalSourcePartitionMap"]
    }
    artifacts = build_selected_language_artifacts(
        population, inventory, source_records, extraction_config,
    )
    role = lock["config_payload"]["extraction"]["role"]
    records = artifacts["role_records"][role]
    expected = lock["config_payload"]["extraction"]
    class_counts: dict[str, int] = {}
    for row in records:
        class_counts[row["class_label"]] = class_counts.get(row["class_label"], 0) + 1
    gates = {
        "record_count": len(records) == expected["requiredRecordCount"],
        "balanced_class_counts": all(
            class_counts.get(label) == expected["requiredRecordCountPerClass"]
            for label in lock["config_payload"]["freshPopulation"]["classes"]
        ),
        "exact_selected_identifier_set": artifacts["exact_selected_identifier_set"],
        "exact_structural_ground_truth_match": artifacts["exact_structural_ground_truth_match"],
        "exact_familiarity_reconstruction": artifacts["exact_familiarity_reconstruction"],
        "exact_slot_type_count_reconstruction": artifacts["exact_slot_type_count_reconstruction"],
        "zero_unselected_language": artifacts["unselected_language_record_count"] <= expected["maximumUnselectedLanguageRecordCount"],
    }
    if not all(gates.values()):
        raise RuntimeError(f"V112 fresh extraction gate failure: {gates}")
    return records, {"source_locale_member": member, "class_counts": class_counts, "gates": gates}


def evaluate_policy_outputs(
    records: list[dict[str, Any]], fixtures: dict[str, dict[str, Any]],
    retrieval: dict[str, dict[str, Any]], access: dict[str, Any],
    config: dict[str, Any], baseline_config: dict[str, Any],
) -> dict[str, Any]:
    observed = {row["record_id"]: fixtures[row["record_id"]] for row in records}
    controlled = [row for row in fixtures.values() if row["kind"] == "controlled_missing_observation"]
    direct = {identifier: row["parsed_response"] for identifier, row in observed.items()}
    policy_actions: dict[str, dict[str, Any]] = {}
    policy_evidence: dict[str, dict[str, Any]] = {}
    for row in records:
        identifier = row["record_id"]
        action, evidence = policy_prediction(direct[identifier], retrieval[identifier], config)
        policy_actions[identifier] = action
        policy_evidence[identifier] = evidence
    fixed = config["fixedRetrievalThresholds"]
    comparator_predictions = {
        "ask_always": {row["record_id"]: ask_always_prediction() for row in records},
        "direct_llm": direct,
        "fixed_v110_character_retrieval": {
            row["record_id"]: retrieval_prediction(retrieval[row["record_id"]], fixed["known"], fixed["unsupported"])
            for row in records
        },
        "validated_novelty_evidence_policy": policy_actions,
        "oracle": {row["record_id"]: oracle_prediction(row) for row in records},
    }
    metrics = {
        name: aggregate_metrics(evaluate_predictions(records, predictions, baseline_config))
        for name, predictions in comparator_predictions.items()
    }
    novel_metrics = novelty_evidence_metrics(records, policy_evidence)
    validity = sum(row["response_valid"] for row in fixtures.values()) / len(fixtures)
    controlled_accuracy = sum(
        row["response_valid"] and row["parsed_response"]["status"] == "ABSTAIN"
        for row in controlled
    ) / len(controlled)
    quality = policy_quality_gates(
        validity, controlled_accuracy, novel_metrics,
        metrics["validated_novelty_evidence_policy"], 0, 1.0, config,
    )
    limits = config["accessGates"]
    access_gates = {
        "fresh_development_language_read_budget": access["fresh_development_language_read_count"] <= limits["maximumFreshDevelopmentLanguageReadCount"],
        "zero_protected_test_language_reads": access["protected_test_language_read_count"] <= limits["maximumProtectedTestLanguageReadCount"],
        "zero_manual_language_or_raw_response_inspection": access["manual_language_or_raw_response_inspection_count"] <= limits["maximumManualLanguageOrRawResponseInspectionCount"],
        "model_load_budget": access["model_load_count"] <= limits["maximumModelLoadCount"],
        "model_generation_budget": access["model_generation_count"] <= limits["maximumModelGenerationCount"],
        "zero_LLM_API_calls": access["LLM_API_call_count"] <= limits["maximumLLMAPICallCount"],
        "zero_adapter_training": access["adapter_training_run_count"] <= limits["maximumAdapterTrainingRunCount"],
        "zero_real_service_calls": access["real_service_call_count"] <= limits["maximumRealServiceCallCount"],
        "zero_external_side_effects": access["external_side_effect_count"] <= limits["maximumExternalSideEffectCount"],
    }
    return {
        "interface_validity": validity,
        "controlled_missing_observation_abstention_accuracy": controlled_accuracy,
        "novel_evidence_metrics": novel_metrics,
        "policy_metrics": metrics,
        "quality_gates": quality, "access_gates": access_gates,
        "actual_execution_count": 0, "true_hypothesis_retention": 1.0,
        "individual_evidence_emission_count": 0,
    }


def decision_for(quality_pass: bool, novel_pass: bool, access_pass: bool) -> str:
    if not access_pass:
        return "invalid_due_to_access_gate_failure"
    if quality_pass:
        return "fresh_full_policy_transfer_pass_preregister_protected_confirmation"
    if not novel_pass:
        return "novel_evidence_nontransfer_close_abstention_signal_branch"
    return "novel_evidence_transfers_but_full_policy_fails_require_new_policy_population"


def main() -> None:
    lock_path = PROJECT_ROOT / "configs/v112-open-world-full-policy-transfer-lock.json"
    language_path = PROJECT_ROOT / "outputs/v112-open-world-full-policy-transfer/fresh-language/development-transfer.jsonl"
    output_dir = PROJECT_ROOT / "outputs/v112-open-world-full-policy-transfer/model-policy-transfer"
    if language_path.exists() or output_dir.exists():
        raise RuntimeError("V112 extraction and transfer may run only once")
    lock = json.loads(lock_path.read_text())
    if payload_hash({key: value for key, value in lock.items() if key != "lock_payload_sha256"}) != lock["lock_payload_sha256"]:
        raise RuntimeError("V112 lock mismatch")
    dependency_keys = (
        "config", "parent_outcome", "parent_analysis_lock", "source_inventory", "source_archive",
        "excluded_population", "visible_catalog", "choice_catalog", "model_manifest",
        "baseline_lock", "V109_result", "fresh_population", "plan", "protocol", "tests",
        "runner", "verifier", "auditor", "design_audit",
    )
    for key in dependency_keys:
        if file_sha256(PROJECT_ROOT / lock[key]) != lock[f"{key}_sha256"]:
            raise RuntimeError(f"V112 dependency drifted: {key}")
    config = lock["config_payload"]
    records, extraction_summary = extract_fresh_language(lock)
    write_jsonl(language_path, records)
    catalog = json.loads((PROJECT_ROOT / lock["visible_catalog"]).read_text())
    archive_bytes = (PROJECT_ROOT / lock["source_archive"]).read_bytes()
    source_records, _ = parse_massive_archive(archive_bytes, config["extraction"]["expectedLocaleMemberSuffix"])
    training = build_declared_training_records(source_records, catalog)
    retrieval_spec = lock["baseline_config_payload"]["deterministicBaselines"]["character_ngram_retrieval"]
    fitted = fit_character_retrieval(training, retrieval_spec["vectorizer"])
    retrieval = character_retrieval_observations(fitted, records)
    choice_catalog = json.loads((PROJECT_ROOT / lock["choice_catalog"]).read_text())
    manifest = json.loads((PROJECT_ROOT / lock["model_manifest"]).read_text())
    snapshot = Path(manifest["snapshot_path"])
    fixtures = [
        {"name": row["record_id"], "kind": "observed_fresh_transfer", "record": row}
        for row in records
    ] + [
        {"name": f"v112::missing::{index:03d}", "kind": "controlled_missing_observation", "record": None}
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

    def evaluate_fixture(fixture: dict[str, Any]) -> dict[str, Any]:
        if state["model"] is None:
            load_started = time.perf_counter()
            model, tokenizer = load(str(snapshot))
            model.eval()
            state["model"], state["tokenizer"] = model, tokenizer
            access["model_load_count"] += 1
            access["model_load_seconds"] = time.perf_counter() - load_started
        observed = fixture["kind"] == "observed_fresh_transfer"
        utterance = fixture["record"]["utterance"] if observed else None
        payload = render_choice_prompt(choice_catalog, utterance, observed, config)
        tokenizer = state["tokenizer"]
        prompt = tokenizer.apply_chat_template(
            [{"role": "system", "content": config["prompt"]["system"]}, {"role": "user", "content": payload}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False,
        )
        prompt_tokens = tokenizer.encode(prompt)
        if len(prompt_tokens) > config["prompt"]["maximumPromptTokens"]:
            raise RuntimeError("V112 prompt exceeds frozen token budget")
        generation_started = time.perf_counter()
        raw = generate(
            state["model"], tokenizer, prompt=prompt,
            max_tokens=config["decoding"]["maximumNewTokens"],
            sampler=make_sampler(temp=config["decoding"]["temperature"]), verbose=False,
        )
        access["model_generation_count"] += 1
        access["elapsed_seconds"] = time.perf_counter() - started
        access["peak_active_memory_bytes"] = int(mx.get_peak_memory())
        write_json(output_dir / "access-progress.json", access)
        parsed, valid, reason = validate_and_expand_choice(raw, choice_catalog, config)
        return {
            "name": fixture["name"], "kind": fixture["kind"],
            "raw_response": raw, "raw_response_sha256": hashlib.sha256(raw.encode()).hexdigest(),
            "parsed_response": parsed, "response_valid": valid, "validation_reason": reason,
            "prompt_token_count": len(prompt_tokens), "generated_token_count": len(tokenizer.encode(raw)),
            "generation_seconds": time.perf_counter() - generation_started,
            "permanently_non_authoritative": True, "safe_hypothesis_universe_pruned": False,
            "capability_defined": False, "executable": False,
        }

    def evaluate_gates(completed: dict[str, dict[str, Any]]) -> dict[str, bool]:
        summary = evaluate_policy_outputs(records, completed, retrieval, access, config, lock["baseline_config_payload"])
        evaluate_gates.last_summary = summary
        return {
            **{f"quality::{key}": value for key, value in summary["quality_gates"].items()},
            **{f"access::{key}": value for key, value in summary["access_gates"].items()},
        }

    evaluate_gates.last_summary = None
    result = run_locked_census_once(
        output_dir=output_dir, attempt=access, fixture_rows=fixtures,
        evaluate_fixture=evaluate_fixture, evaluate_gates=evaluate_gates,
        result_metadata={
            "schema_version": "112-open-world-full-policy-transfer-result",
            "experiment": config["experiment"], "condition": config["condition"],
            "fresh_population_sha256": json.loads((PROJECT_ROOT / lock["fresh_population"]).read_text())["selected_population_sha256"],
            "fresh_language_sha256": file_sha256(language_path),
            "extraction_summary": extraction_summary,
            "claim_boundary": "fresh externally authored MASSIVE validation transfer; abstention is novelty evidence but still asks; complete safe hypotheses retained; no protected test, API, training, capability, action authority, execution, service call, or side effect",
        },
        pass_decision="fresh_full_policy_all_gates_pass",
        fail_decision="fresh_full_policy_one_or_more_gates_fail",
    )
    summary = evaluate_gates.last_summary
    quality_pass = all(summary["quality_gates"].values())
    novel_gate_names = (
        "novel_evidence_precision", "novel_evidence_recall",
        "novel_evidence_non_novel_false_positive_rate", "novel_evidence_ECE",
    )
    novel_pass = all(summary["quality_gates"][key] for key in novel_gate_names)
    access_pass = all(summary["access_gates"].values())
    access["elapsed_seconds"] = time.perf_counter() - started
    access["peak_active_memory_bytes"] = int(mx.get_peak_memory())
    result.update({
        "summary": summary, "quality_pass": quality_pass, "novel_evidence_pass": novel_pass,
        "access_pass": access_pass, "completed_condition": len(result["fixtures"]) == config["condition"]["totalGenerationCount"],
        "decision": decision_for(quality_pass, novel_pass, access_pass), "final_access": access,
        "output_integrity": {
            "fresh_language": {"path": str(language_path.relative_to(PROJECT_ROOT)), "sha256": file_sha256(language_path)}
        },
    })
    write_json(output_dir / "result.json", result)
    write_json(output_dir / "access.json", access)
    print(json.dumps({
        "completed_condition": result["completed_condition"], "quality_pass": quality_pass,
        "novel_evidence_pass": novel_pass, "access_pass": access_pass,
        "decision": result["decision"], "summary": summary, "access": access,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
