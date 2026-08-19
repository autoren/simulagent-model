#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import importlib.metadata as metadata
import json
from pathlib import Path
from typing import Any

from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT
from v109_open_world_typed_choice import compile_choice_catalog


def payload_hash(value: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def valid_lock(payload: dict[str, Any]) -> bool:
    return payload_hash({key: value for key, value in payload.items() if key != "lock_payload_sha256"}) == payload.get("lock_payload_sha256")


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def main() -> None:
    config_path = PROJECT_ROOT / "configs/v109-open-world-typed-choice.json"
    parent_path = PROJECT_ROOT / "configs/v108-open-world-interface-forensics-outcome-lock.json"
    plan_path = PROJECT_ROOT / "docs/v109-open-world-typed-choice-plan.md"
    protocol_path = PROJECT_ROOT / "python/v109_open_world_typed_choice.py"
    tests_path = PROJECT_ROOT / "python/test_v109_open_world_typed_choice.py"
    runner_path = PROJECT_ROOT / "python/run_v109_open_world_typed_choice.py"
    verifier_path = PROJECT_ROOT / "python/verify_and_freeze_v109_typed_choice_outcome.py"
    auditor_path = PROJECT_ROOT / "python/audit_and_freeze_v109_typed_choice_implementation.py"
    harness_path = PROJECT_ROOT / "python/locked_census_harness.py"
    audit_path = PROJECT_ROOT / "outputs/v109-open-world-typed-choice/implementation-audit.json"
    choice_path = PROJECT_ROOT / "outputs/v109-open-world-typed-choice/interface/choice-catalog.json"
    lock_path = PROJECT_ROOT / "configs/v109-open-world-typed-choice-implementation-lock.json"
    model_output = PROJECT_ROOT / "outputs/v109-open-world-typed-choice/holdback-evaluation"
    if audit_path.exists() or choice_path.exists() or lock_path.exists() or model_output.exists():
        raise RuntimeError("V109 implementation is already frozen or materialized")

    config = json.loads(config_path.read_text())
    parent = json.loads(parent_path.read_text())
    v108_lock = json.loads((PROJECT_ROOT / parent["diagnostic_lock"]).read_text())
    v107_outcome = json.loads((PROJECT_ROOT / v108_lock["parent_model_outcome"]).read_text())
    v107_lock = json.loads((PROJECT_ROOT / v107_outcome["implementation_lock"]).read_text())
    baseline_outcome = json.loads((PROJECT_ROOT / v107_lock["parent_baseline_outcome"]).read_text())
    baseline_lock = json.loads((PROJECT_ROOT / baseline_outcome["benchmark_lock"]).read_text())
    catalog_path = PROJECT_ROOT / baseline_lock["visible_catalog"]
    catalog = json.loads(catalog_path.read_text())
    choices = compile_choice_catalog(catalog, config)
    membership_path = PROJECT_ROOT / baseline_outcome["development_split_membership"]
    membership = json.loads(membership_path.read_text())
    calibration_ids = {row["record_id"] for row in membership["membership"] if row["subset"] == "calibration"}
    evaluation_ids = {row["record_id"] for row in membership["membership"] if row["subset"] == "evaluation"}
    v107_result_path = PROJECT_ROOT / v107_outcome["result"]
    v107_result = json.loads(v107_result_path.read_text())
    v107_observed_ids = {
        identifier for identifier, row in v107_result["fixtures"].items()
        if row["kind"] == "observed_evaluation"
    }
    manifest_path = PROJECT_ROOT / config["modelManifest"]
    manifest = json.loads(manifest_path.read_text())
    snapshot = Path(manifest["snapshot_path"])
    condition = config["condition"]
    runtime_versions = {
        package: metadata.version(package)
        for package in ("mlx", "mlx-lm", "huggingface-hub", "transformers")
    }
    choice_counts: dict[str, int] = {}
    for row in choices["choices"]:
        choice_counts[row["kind"]] = choice_counts.get(row["kind"], 0) + 1
    choice_ids = [row["choice_id"] for row in choices["choices"]]
    checks = {
        "V108_exactly_authorizes_fresh_constrained_development_study": bool(
            valid_lock(parent) and parent["outcome"]["passed"]
            and parent["outcome"]["format_dominance_passed"]
            and parent["authorization"]["preregister_fresh_constrained_typed_interface_development_study"]
            and not parent["authorization"]["read_protected_test_or_run_model_before_fresh_lock"]
        ),
        "holdback_membership_is_balanced_and_disjoint_from_V107_generation": bool(
            len(calibration_ids) == 128 and len(evaluation_ids) == 128
            and calibration_ids.isdisjoint(evaluation_ids)
            and v107_observed_ids == evaluation_ids
            and calibration_ids.isdisjoint(v107_observed_ids)
            and config["corpus"]["excludeV107EvaluationRecords"]
            and config["corpus"]["excludeProtectedTestRecords"]
        ),
        "single_choice_catalog_is_complete_unique_and_non_aliasing": bool(
            len(choice_ids) == config["typedChoiceInterface"]["requiredChoiceCount"] == 17
            and len(choice_ids) == len(set(choice_ids))
            and choice_counts == {"KNOWN": 12, "NOVEL": 3, "UNSUPPORTED": 1, "ABSTAIN": 1}
            and config["typedChoiceInterface"]["singleMachineAcceptedIdentifierField"] == "choice_id"
            and config["typedChoiceInterface"]["outputKeys"] == ["choice_id", "confidence"]
            and not config["typedChoiceInterface"]["acceptAliases"]
            and not config["typedChoiceInterface"]["acceptQualifiedIntentOutsideChoiceId"]
        ),
        "pinned_local_snapshot_manifest_is_exact_and_present": bool(
            file_sha256(manifest_path) == config["modelManifestSha256"]
            and manifest["repository"] == condition["repository"]
            and manifest["revision"] == condition["revision"]
            and manifest["quantization_bits"] == condition["quantizationBits"]
            and manifest["weight_bytes"] == condition["weightBytes"]
            and snapshot.is_dir() and snapshot.name == condition["revision"]
        ),
        "one_192_generation_no_retry_condition_is_frozen": bool(
            config["corpus"]["observedDevelopmentHoldbackCount"] == 128
            and config["corpus"]["controlledMissingObservationCount"] == 64
            and config["corpus"]["totalGenerationCount"] == 192
            and config["prompt"]["demonstrationCount"] == 0
            and config["decoding"]["temperature"] == 0.0
            and config["decoding"]["samplesPerRecord"] == 1
            and not config["decoding"]["retryOnMalformedOutput"]
            and not config["decoding"]["enableThinking"]
        ),
        "interface_semantic_access_gates_and_decision_paths_are_frozen": bool(
            len(config["interfaceGates"]) == 5 and len(config["semanticGates"]) == 13
            and config["semanticGates"]["maximumFalseKnownAcceptanceRate"] == 0.10
            and config["semanticGates"]["maximumRegretAboveAskAlways"] == 0.25
            and not config["decisionRule"]["passAuthorizesProtectedTest"]
            and not config["decisionRule"]["passAuthorizesAPITrainingPlanningOrExecution"]
        ),
        "model_remains_non_authoritative_and_complete_hypotheses_remain": bool(
            config["authorityBoundary"]["permanentlyNonAuthoritative"]
            and config["authorityBoundary"]["completeSafeHypothesisUniverseRetained"]
            and not config["authorityBoundary"]["modelDefinesOrPrunesCapabilities"]
            and not config["authorityBoundary"]["modelUpdatesPosterior"]
            and not config["authorityBoundary"]["modelSelectsAction"]
            and not config["authorityBoundary"]["modelExecutesTool"]
            and config["authorityBoundary"]["realExecutionCount"] == 0
        ),
        "frozen_inputs_code_runtime_and_output_absence_hold": bool(
            file_sha256(catalog_path) == baseline_lock["visible_catalog_sha256"]
            and file_sha256(membership_path) == baseline_outcome["development_split_membership_sha256"]
            and all(path.is_file() for path in (
                plan_path, protocol_path, tests_path, runner_path, verifier_path, auditor_path, harness_path,
            ))
            and runtime_versions["mlx-lm"] == "0.31.3" and not model_output.exists()
        ),
    }
    passed = all(checks.values())
    audit = {
        "schema_version": "109-open-world-typed-choice-implementation-audit",
        "experiment": "v109_open_world_typed_choice_implementation_audit",
        "passed": passed,
        "decision": "freeze_and_authorize_one_holdback_typed_choice_run" if passed else "reject_V109_model_inference",
        "checks": checks, "choice_counts": choice_counts, "runtime_versions": runtime_versions,
        "access": {
            "development_language_read_count": 0, "protected_test_language_read_count": 0,
            "manual_utterance_inspection_count": 0, "model_load_count": 0,
            "model_generation_count": 0, "LLM_API_call_count": 0,
            "adapter_training_run_count": 0, "real_service_call_count": 0,
            "external_side_effect_count": 0,
        },
    }
    write_json(audit_path, audit)
    if not passed:
        print(json.dumps(audit, indent=2, sort_keys=True))
        raise SystemExit(1)
    write_json(choice_path, choices)
    dependencies = {
        "config": config_path, "parent_forensics_outcome": parent_path,
        "forensics_lock": PROJECT_ROOT / parent["diagnostic_lock"],
        "V107_outcome": PROJECT_ROOT / v108_lock["parent_model_outcome"],
        "V107_implementation_lock": PROJECT_ROOT / v107_outcome["implementation_lock"],
        "V107_result": v107_result_path,
        "baseline_outcome": PROJECT_ROOT / v107_lock["parent_baseline_outcome"],
        "baseline_lock": PROJECT_ROOT / baseline_outcome["benchmark_lock"],
        "development_membership": membership_path,
        "development_language": PROJECT_ROOT / baseline_lock["development_language"],
        "visible_catalog": catalog_path,
        "controlled_identifiers": PROJECT_ROOT / baseline_lock["controlled_identifiers"],
        "model_manifest": manifest_path, "choice_catalog": choice_path,
        "plan": plan_path, "protocol": protocol_path, "tests": tests_path,
        "runner": runner_path, "verifier": verifier_path, "auditor": auditor_path,
        "census_harness": harness_path, "implementation_audit": audit_path,
    }
    lock: dict[str, Any] = {
        "schema_version": "109-open-world-typed-choice-implementation-lock",
        "experiment": "v109_open_world_typed_choice_implementation_lock",
        "config_payload": config,
        "baseline_config_payload": baseline_lock["config_payload"],
        "interface_config_payload": v107_lock["interface_config_payload"],
        "runtime_versions": runtime_versions,
        "authorization": {
            "modify_corpus_choice_catalog_prompt_model_decoding_metrics_costs_gates_or_decisions": False,
            "run_one_local_holdback_shadow_condition_once": True,
            "maximum_model_load_count": 1, "maximum_model_generation_count": 192,
            "retry_failed_malformed_or_negative_fixture": False,
            "read_protected_test_language": False,
            "manually_inspect_source_or_model_language": False,
            "run_API_model_or_train_adapter": False,
            "prune_safe_hypotheses_or_grant_model_belief_action_or_execution_authority": False,
            "perform_real_service_call_or_external_side_effect": False,
        },
    }
    for key, path in dependencies.items():
        lock[key] = str(path.relative_to(PROJECT_ROOT))
        lock[f"{key}_sha256"] = file_sha256(path)
    lock["lock_payload_sha256"] = payload_hash(lock)
    write_json(lock_path, lock)
    print(json.dumps(audit, indent=2, sort_keys=True))
    print(json.dumps({"lock": str(lock_path.relative_to(PROJECT_ROOT)), "sha256": file_sha256(lock_path)}, indent=2))


if __name__ == "__main__":
    main()
