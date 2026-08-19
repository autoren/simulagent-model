#!/usr/bin/env python3
"""Verify V91 implementation and planner invariance before local-model inference."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT
from v91_rank_only_protocol import parse_and_complete, verify_v79_permutation_invariance


def payload_hash(value: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def main() -> None:
    design_path = PROJECT_ROOT / "configs/v91-rank-only-design-lock.json"
    seal_path = PROJECT_ROOT / "data/v91-rank-only/corpus-seal.json"
    protocol_path = PROJECT_ROOT / "python/v91_rank_only_protocol.py"
    tests_path = PROJECT_ROOT / "python/test_v91_rank_only_protocol.py"
    runner_path = PROJECT_ROOT / "python/run_v91_rank_only_mlx.py"
    census_path = PROJECT_ROOT / "python/locked_census_harness.py"
    auditor_path = PROJECT_ROOT / "python/audit_and_freeze_v91_implementation.py"
    audit_path = PROJECT_ROOT / "outputs/v91-rank-only/implementation-audit.json"
    planner_result_path = PROJECT_ROOT / "outputs/v91-rank-only/planner-invariance.json"
    lock_path = PROJECT_ROOT / "configs/v91-rank-only-implementation-lock.json"
    evaluation_path = PROJECT_ROOT / "outputs/v91-rank-only/evaluation"
    if (
        audit_path.exists()
        or planner_result_path.exists()
        or lock_path.exists()
        or evaluation_path.exists()
    ):
        raise RuntimeError("V91 implementation is already audited, frozen, or evaluated")

    design = json.loads(design_path.read_text())
    design_payload = {
        key: value for key, value in design.items() if key != "lock_payload_sha256"
    }
    seal = json.loads(seal_path.read_text())
    seal_payload = {
        key: value for key, value in seal.items() if key != "lock_payload_sha256"
    }
    config = design["config_payload"]
    corpus_path = PROJECT_ROOT / seal["corpus"]
    records = [
        json.loads(line) for line in corpus_path.read_text().splitlines() if line
    ]
    model = config["modelCondition"]
    manifest_path = PROJECT_ROOT / model["reuseManifest"]
    manifest = json.loads(manifest_path.read_text())
    snapshot = Path(manifest["snapshot_path"])

    planner_outcome = json.loads(
        (PROJECT_ROOT / design["planner_outcome_lock"]).read_text()
    )
    planner_impl_path = PROJECT_ROOT / planner_outcome["implementation_lock"]
    if file_sha256(planner_impl_path) != planner_outcome["implementation_lock_sha256"]:
        raise RuntimeError("V91 frozen V79 planner implementation lock drifted")
    planner_impl = json.loads(planner_impl_path.read_text())
    planner_result = verify_v79_permutation_invariance(
        planner_impl["resolved_config_payload"]
    )
    planner_result_path.write_text(
        json.dumps(planner_result, indent=2, sort_keys=True) + "\n"
    )

    malformed = parse_and_complete(
        "not json", ["BookAppointment", "FindProvider", "NONE"]
    )
    adversarial = parse_and_complete(
        json.dumps({"intent_priority": ["NONE", "unknown", "NONE"]}),
        ["BookAppointment", "FindProvider", "NONE"],
    )
    checks = {
        "design_and_corpus_locks_are_exact": bool(
            payload_hash(design_payload) == design["lock_payload_sha256"]
            and payload_hash(seal_payload) == seal["lock_payload_sha256"]
            and seal["design_lock_sha256"] == file_sha256(design_path)
            and file_sha256(corpus_path) == seal["corpus_sha256"]
            and all(
                file_sha256(PROJECT_ROOT / design[key])
                == design[f"{key}_sha256"]
                for key in (
                    "design",
                    "source_outcome_lock",
                    "source_inventory",
                    "parent_model_decision_lock",
                    "planner_outcome_lock",
                    "plan",
                    "protocol",
                    "tests",
                    "auditor",
                    "builder",
                )
            )
        ),
        "sealed_population_is_exact_balanced_unique_and_uninspected": bool(
            len(records) == seal["record_count"] == 64
            and seal["dialogue_count"] == 64
            and len({record["source_dialogue_id"] for record in records}) == 64
            and sum(record["gold_intent"] == "NONE" for record in records) == 32
            and sum(record["gold_intent"] != "NONE" for record in records) == 32
            and seal["manual_utterance_inspection_count"] == 0
        ),
        "reused_4B_snapshot_manifest_is_exact_without_new_download_or_load": bool(
            file_sha256(manifest_path) == model["reuseManifestFileSha256"]
            and manifest["manifest_sha256"]
            == model["reuseManifestPayloadSha256"]
            and manifest["repository"] == model["repository"]
            and manifest["revision"] == model["revision"]
            and manifest["weight_bytes"] == model["weightBytes"]
            and snapshot.is_dir()
            and snapshot.name == model["revision"]
            and seal["new_model_weight_download_count"] == 0
            and seal["model_load_count"] == 0
            and seal["model_generation_count"] == 0
        ),
        "malformed_unknown_duplicate_and_omitted_items_cannot_prune_or_drop_NONE": bool(
            malformed["completed_priority"]
            == ["BookAppointment", "FindProvider", "NONE"]
            and malformed["canonical_complete_set"]
            and malformed["canonical_NONE_retained"]
            and adversarial["completed_priority"]
            == ["NONE", "BookAppointment", "FindProvider"]
            and adversarial["canonical_complete_set"]
            and adversarial["canonical_NONE_retained"]
        ),
        "all_V79_hypothesis_permutations_preserve_exact_policy_and_certification": bool(
            planner_result["fixture_count"]
            == config["plannerInvariance"]["requiredFixtureCount"]
            and planner_result["permutation_count"]
            == config["plannerInvariance"]["requiredFixtureCount"]
            * config["plannerInvariance"][
                "requiredHypothesisPermutationCountPerFixture"
            ]
            and planner_result["invariance_rate"] == 1.0
            and planner_result["action_mismatch_count"] == 0
            and planner_result["maximum_absolute_value_error"]
            <= config["plannerInvariance"]["maximumAbsoluteValueError"]
            and planner_result["execution_certificate_violation_count"] == 0
            and planner_result["model_output_access_count"] == 0
        ),
        "runner_protocol_tests_census_and_auditor_are_frozen": all(
            path.is_file()
            for path in (
                protocol_path,
                tests_path,
                runner_path,
                census_path,
                auditor_path,
            )
        ),
        "no_evaluation_artifact_exists_before_inference_lock": not evaluation_path.exists(),
        "implementation_grants_no_pruning_belief_action_API_training_or_execution": bool(
            not design["authorization"]["prune_or_early_stop_search"]
            and not design["authorization"][
                "grant_model_belief_action_or_execution_authority"
            ]
            and not design["authorization"]["run_API_model_or_train_adapter"]
            and not design["authorization"][
                "perform_real_service_call_or_external_side_effect"
            ]
        ),
    }
    passed = all(checks.values())
    try:
        import importlib.metadata as metadata

        runtime_versions = {
            package: metadata.version(package)
            for package in ("mlx", "mlx-lm", "huggingface-hub", "transformers")
        }
    except Exception as error:  # pragma: no cover
        runtime_versions = {"error": str(error)}
        passed = False
    audit = {
        "schema_version": "91-rank-only-implementation-audit",
        "experiment": "v91_rank_only_implementation_audit",
        "passed": passed,
        "decision": (
            "freeze_and_authorize_one_local_rank_only_census"
            if passed
            else "reject_V91_inference"
        ),
        "checks": checks,
        "planner_invariance": planner_result,
        "runtime_versions": runtime_versions,
        "access": {
            "selected_source_language_record_count": len(records),
            "manual_utterance_inspection_count": 0,
            "new_model_weight_download_count": 0,
            "model_load_count": 0,
            "model_generation_count": 0,
            "LLM_API_call_count": 0,
            "adapter_training_run_count": 0,
            "real_service_call_count": 0,
            "external_side_effect_count": 0,
        },
    }
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    if not passed:
        print(json.dumps(audit, indent=2, sort_keys=True))
        raise SystemExit(1)

    lock = {
        "schema_version": "91-rank-only-implementation-lock",
        "experiment": "v91_rank_only_implementation_lock",
        "design_lock": str(design_path.relative_to(PROJECT_ROOT)),
        "design_lock_sha256": file_sha256(design_path),
        "config_payload": config,
        "corpus_seal": str(seal_path.relative_to(PROJECT_ROOT)),
        "corpus_seal_sha256": file_sha256(seal_path),
        "corpus": seal["corpus"],
        "corpus_sha256": seal["corpus_sha256"],
        "protocol": str(protocol_path.relative_to(PROJECT_ROOT)),
        "protocol_sha256": file_sha256(protocol_path),
        "tests": str(tests_path.relative_to(PROJECT_ROOT)),
        "tests_sha256": file_sha256(tests_path),
        "runner": str(runner_path.relative_to(PROJECT_ROOT)),
        "runner_sha256": file_sha256(runner_path),
        "census_harness": str(census_path.relative_to(PROJECT_ROOT)),
        "census_harness_sha256": file_sha256(census_path),
        "model_manifest": str(manifest_path.relative_to(PROJECT_ROOT)),
        "model_manifest_sha256": file_sha256(manifest_path),
        "model_snapshot_path": str(snapshot),
        "planner_outcome_lock": design["planner_outcome_lock"],
        "planner_outcome_lock_sha256": design["planner_outcome_lock_sha256"],
        "planner_implementation_lock": str(planner_impl_path.relative_to(PROJECT_ROOT)),
        "planner_implementation_lock_sha256": file_sha256(planner_impl_path),
        "planner_invariance_result": str(
            planner_result_path.relative_to(PROJECT_ROOT)
        ),
        "planner_invariance_result_sha256": file_sha256(planner_result_path),
        "implementation_auditor": str(auditor_path.relative_to(PROJECT_ROOT)),
        "implementation_auditor_sha256": file_sha256(auditor_path),
        "implementation_audit": str(audit_path.relative_to(PROJECT_ROOT)),
        "implementation_audit_sha256": file_sha256(audit_path),
        "runtime_versions": runtime_versions,
        "authorization": {
            "modify_corpus_model_prompt_protocol_decoding_controls_gates_or_decisions": False,
            "run_one_local_rank_only_census_once": True,
            "maximum_model_load_count": 1,
            "maximum_model_generation_count": 64,
            "rerun_failed_malformed_or_negative_records": False,
            "download_new_model_weights": False,
            "prune_or_early_stop_search": False,
            "mutate_authoritative_state_or_update_belief_from_model": False,
            "grant_model_action_or_execution_authority": False,
            "run_API_model_or_train_adapter": False,
            "manually_inspect_source_language": False,
            "perform_real_service_call_or_external_side_effect": False
        },
    }
    lock["lock_payload_sha256"] = payload_hash(lock)
    lock_path.write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n")
    print(json.dumps(audit, indent=2, sort_keys=True))
    print(
        json.dumps(
            {"lock": str(lock_path.relative_to(PROJECT_ROOT)), "sha256": file_sha256(lock_path)},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
