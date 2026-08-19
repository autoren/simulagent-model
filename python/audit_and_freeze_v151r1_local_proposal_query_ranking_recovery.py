#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from audit_and_freeze_v122_prequery_signal_inventory import payload_hash, valid_lock, write_json
from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT
from v151r1_local_proposal_query_ranking_recovery import derive_partition, recovery_evaluation_config


def main() -> None:
    config_path = PROJECT_ROOT / "configs/v151r1-local-proposal-query-ranking-recovery.json"
    plan_path = PROJECT_ROOT / "docs/v151r1-local-proposal-query-ranking-recovery-plan.md"
    protocol_path = PROJECT_ROOT / "python/v151r1_local_proposal_query_ranking_recovery.py"
    tests_path = PROJECT_ROOT / "python/test_v151r1_local_proposal_query_ranking_recovery.py"
    auditor_path = PROJECT_ROOT / "python/audit_and_freeze_v151r1_local_proposal_query_ranking_recovery.py"
    runner_path = PROJECT_ROOT / "python/run_v151r1_local_proposal_query_ranking_recovery.py"
    verifier_path = PROJECT_ROOT / "python/verify_and_freeze_v151r1_local_proposal_query_ranking_recovery_outcome.py"
    preregistration_dir = PROJECT_ROOT / "outputs/v151r1-local-proposal-query-ranking-recovery/preregistration"
    audit_path = PROJECT_ROOT / "outputs/v151r1-local-proposal-query-ranking-recovery/design-audit.json"
    lock_path = PROJECT_ROOT / "configs/v151r1-local-proposal-query-ranking-recovery-lock.json"
    if preregistration_dir.exists() or audit_path.exists() or lock_path.exists():
        raise RuntimeError("V151r1 already preregistered")

    recovery = json.loads(config_path.read_text())
    parent_path = PROJECT_ROOT / recovery["parentV151AnalysisLock"]
    parent = json.loads(parent_path.read_text())
    base = parent["config_payload"]
    public_path = PROJECT_ROOT / parent["development_public_fixtures"]
    public = json.loads(public_path.read_text())
    original_dir = PROJECT_ROOT / "outputs/v151-local-proposal-query-ranking/model-realization"
    prior_access_path = original_dir / "access-progress.json"
    prior_attempt_path = original_dir / "attempt.json"
    prior_access = json.loads(prior_access_path.read_text())
    persisted_paths = sorted((original_dir / "raw-fixtures").glob("*.json"))
    partition = derive_partition(public, persisted_paths, prior_access, recovery)
    manifest_records = []
    for ordinal, path in enumerate(persisted_paths):
        manifest_records.append(
            {
                "ordinal": ordinal,
                "fixture_id": partition["persisted_fixture_ids"][ordinal],
                "path": str(path.relative_to(PROJECT_ROOT)),
                "sha256": file_sha256(path),
            }
        )
    process_running = False
    try:
        import subprocess

        process_running = bool(
            subprocess.run(
                ["pgrep", "-f", "[r]un_v151_local_proposal_query_ranking.py"],
                capture_output=True,
                text=True,
                check=False,
            ).stdout.strip()
        )
    except OSError:
        process_running = False

    parent_dependencies = [key for key in parent if not key.endswith("_sha256") and f"{key}_sha256" in parent]
    zero_keys = (
        "closed_answer_model_generation_count",
        "evaluation_fixture_model_generation_count",
        "retry_count",
        "manual_raw_response_inspection_count",
        "persisted_raw_response_count",
        "API_call_count",
        "training_run_count",
        "real_service_call_count",
        "external_side_effect_count",
        "actual_execution_count",
    )
    evaluation_config = recovery_evaluation_config(base, recovery)
    checks = {
        "V151_analysis_lock_and_dependencies_exact": bool(
            valid_lock(parent)
            and all(file_sha256(PROJECT_ROOT / parent[key]) == parent[f"{key}_sha256"] for key in parent_dependencies)
            and parent["authorization"]["run_exact_single_pinned_local_development_realization"]
        ),
        "original_process_stopped_without_aggregate_result_or_failure": bool(
            not process_running
            and not (original_dir / "result.json").exists()
            and not (original_dir / "access.json").exists()
            and not (original_dir / "failure.json").exists()
        ),
        "exact_interrupted_partition": bool(
            len(partition["persisted_fixture_ids"]) == recovery["interruption"]["requiredPersistedFixtureCount"]
            and prior_access["model_generation_count"] == recovery["interruption"]["requiredAttemptedGenerationCount"]
            and len(partition["never_started_fixture_ids"]) == recovery["interruption"]["requiredNeverStartedFixtureCount"]
            and len(partition["all_fixture_ids"]) == recovery["interruption"]["requiredFinalFixtureCount"]
        ),
        "prior_single_load_and_all_prohibited_counters_zero": bool(
            prior_access["model_load_count"] == recovery["interruption"]["requiredPriorModelLoadCount"]
            and prior_access["tokenizer_load_count"] == recovery["interruption"]["requiredPriorModelLoadCount"]
            and prior_access["maximum_generation_count_per_fixture"] == 1
            and all(prior_access[key] == 0 for key in zero_keys)
        ),
        "recovery_never_regenerates_persisted_or_interrupted_fixture": bool(
            not recovery["interruption"]["regeneratePersistedOrInterruptedFixture"]
            and set(partition["persisted_fixture_ids"]).isdisjoint(partition["never_started_fixture_ids"])
            and partition["interrupted_fixture_id"] not in partition["never_started_fixture_ids"]
        ),
        "semantic_contract_unchanged_except_recovery_access_and_decision": bool(
            evaluation_config["model"] == base["model"]
            and evaluation_config["prompt"] == base["prompt"]
            and evaluation_config["population"] == base["population"]
            and evaluation_config["qualificationGates"] == base["qualificationGates"]
            and evaluation_config["fallbackQueryRanking"] == base["fallbackQueryRanking"]
            and evaluation_config["accessGates"] == recovery["recoveryAccessGates"]
        ),
        "one_recovery_load_and_final_access_counts_exact": bool(
            recovery["interruption"]["maximumRecoveryModelLoadCount"] == 1
            and recovery["interruption"]["requiredFinalModelLoadCount"] == 2
            and recovery["recoveryAccessGates"]["maximumModelGenerationCount"] == 96
            and recovery["recoveryAccessGates"]["maximumGenerationCountPerFixture"] == 1
        ),
        "no_semantic_output_inspection_or_mining_authorized": bool(
            not recovery["interruption"]["inspectOrMinePersistedSemanticOutputsBeforeRecoveryLock"]
            and not recovery["decisionRule"]["outcomeAuthorizesImmediateEvaluationRun"]
            and not recovery["decisionRule"]["outcomeAuthorizesRetryRerunPromptChangeThresholdFitAPITrainingInductionAuthorityActionOrExecution"]
        ),
        "required_recovery_files_exist": all(
            path.is_file()
            for path in (config_path, plan_path, protocol_path, tests_path, auditor_path, runner_path, verifier_path)
        ),
    }
    passed = all(checks.values())
    preregistration_dir.mkdir(parents=True, exist_ok=False)
    manifest_path = preregistration_dir / "retained-partial-manifest.json"
    manifest = {
        "schema_version": "151r1-retained-partial-manifest",
        "semantic_output_fields_read_during_design_audit": False,
        "persisted_records": manifest_records,
        "interrupted_fixture_id": partition["interrupted_fixture_id"],
        "never_started_fixture_ids": partition["never_started_fixture_ids"],
        "all_fixture_ids": partition["all_fixture_ids"],
        "prior_access_sha256": file_sha256(prior_access_path),
        "prior_attempt_sha256": file_sha256(prior_attempt_path),
    }
    write_json(manifest_path, manifest)
    audit = {
        "schema_version": "151r1-local-proposal-query-ranking-recovery-design-audit",
        "experiment": recovery["experiment"],
        "passed": passed,
        "checks": checks,
        "persisted_fixture_count": len(manifest_records),
        "interrupted_fixture_count": 1,
        "never_started_fixture_count": len(partition["never_started_fixture_ids"]),
        "persisted_semantic_output_inspection_count": 0,
        "model_load_count_during_recovery_design": 0,
        "model_generation_count_during_recovery_design": 0,
        "decision": "authorize_exact_V151r1_no_retry_recovery" if passed else "close_V151r1_before_recovery",
    }
    write_json(audit_path, audit)
    if not passed:
        print(json.dumps(audit, indent=2, sort_keys=True))
        raise SystemExit(1)

    dependencies = {
        "config": config_path,
        "parent_analysis_lock": parent_path,
        "development_public_fixtures": public_path,
        "development_hidden_fixtures": PROJECT_ROOT / parent["development_hidden_fixtures"],
        "development_answer_metadata": PROJECT_ROOT / parent["development_answer_metadata"],
        "interaction_catalog": PROJECT_ROOT / parent["interaction_catalog"],
        "witness_config": PROJECT_ROOT / parent["witness_config"],
        "oracle_config": PROJECT_ROOT / parent["oracle_config"],
        "model_manifest": PROJECT_ROOT / parent["model_manifest"],
        "prior_access_progress": prior_access_path,
        "prior_attempt": prior_attempt_path,
        "retained_partial_manifest": manifest_path,
        "plan": plan_path,
        "protocol": protocol_path,
        "tests": tests_path,
        "auditor": auditor_path,
        "runner": runner_path,
        "verifier": verifier_path,
        "design_audit": audit_path,
    }
    lock: dict[str, Any] = {
        "schema_version": "151r1-local-proposal-query-ranking-recovery-lock",
        "experiment": recovery["experiment"],
        "recovery_config_payload": recovery,
        "base_V151_config_payload": base,
        "authorization": {
            "retain_exact_58_hash_locked_outputs": True,
            "assign_interrupted_fixture_deterministic_invalid_fail_closed_output": True,
            "generate_exactly_37_never_started_fixtures_once": True,
            "regenerate_persisted_or_interrupted_fixture": False,
            "inspect_mine_modify_retry_rerun_reprompt_tune_or_threshold_fit": False,
            "generate_on_closed_answer_or_V149_evaluation_fixtures": False,
            "run_API_training_induction_authority_action_or_execution": False,
        },
    }
    for key, path in dependencies.items():
        lock[key] = str(path.relative_to(PROJECT_ROOT))
        lock[f"{key}_sha256"] = file_sha256(path)
    lock["lock_payload_sha256"] = payload_hash(lock)
    write_json(lock_path, lock)
    print(json.dumps({"passed": passed, "checks": checks, "decision": audit["decision"]}, indent=2, sort_keys=True))
    print(json.dumps({"lock": str(lock_path.relative_to(PROJECT_ROOT)), "sha256": file_sha256(lock_path)}, indent=2))


if __name__ == "__main__":
    main()
