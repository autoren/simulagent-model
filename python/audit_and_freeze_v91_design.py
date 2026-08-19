#!/usr/bin/env python3
"""Audit and freeze V91 before selected language extraction or model access."""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from typing import Any

from build_v91_rank_only_corpus import structurally_select
from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT


def payload_hash(value: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def main() -> None:
    design_path = PROJECT_ROOT / "configs/v91-rank-only-design.json"
    source_path = PROJECT_ROOT / "configs/v91-rank-only-source-outcome-lock.json"
    parent_model_path = PROJECT_ROOT / "configs/v90-capacity-generation-outcome-lock.json"
    planner_path = PROJECT_ROOT / "configs/v79-terminal-utility-outcome-lock.json"
    plan_path = PROJECT_ROOT / "docs/v91-rank-only-plan.md"
    protocol_path = PROJECT_ROOT / "python/v91_rank_only_protocol.py"
    tests_path = PROJECT_ROOT / "python/test_v91_rank_only_protocol.py"
    auditor_path = PROJECT_ROOT / "python/audit_and_freeze_v91_design.py"
    builder_path = PROJECT_ROOT / "python/build_v91_rank_only_corpus.py"
    audit_path = PROJECT_ROOT / "outputs/v91-rank-only/design-audit.json"
    lock_path = PROJECT_ROOT / "configs/v91-rank-only-design-lock.json"
    corpus_root = PROJECT_ROOT / "data/v91-rank-only"
    if audit_path.exists() or lock_path.exists() or corpus_root.exists():
        raise RuntimeError("V91 design is already frozen or materialized")

    config = json.loads(design_path.read_text())
    source = json.loads(source_path.read_text())
    parent_model = json.loads(parent_model_path.read_text())
    planner = json.loads(planner_path.read_text())
    source_payload = {
        key: value for key, value in source.items() if key != "lock_payload_sha256"
    }
    parent_model_payload = {
        key: value
        for key, value in parent_model.items()
        if key != "lock_payload_sha256"
    }
    planner_payload = {
        key: value for key, value in planner.items() if key != "lock_payload_sha256"
    }
    inventory = json.loads((PROJECT_ROOT / source["inventory"]).read_text())
    selected = structurally_select(config, inventory["record_index"])
    strata = Counter((row["service"], row["active_intent"]) for row in selected)
    requested = Counter(
        {
            (item["service"], item["activeIntent"]): item["count"]
            for item in config["population"]["strata"]
        }
    )
    model = config["modelCondition"]
    quality = config["qualityGates"]
    access = config["accessGates"]
    stage = config["stageAuthorization"]
    planner_audit = json.loads((PROJECT_ROOT / planner["audit"]).read_text())
    checks = {
        "positive_fresh_source_outcome_is_exact_and_authorizes_rank_design": bool(
            payload_hash(source_payload) == source["lock_payload_sha256"]
            and source["outcome"]["passed"]
            and source["authorization"][
                "preregister_fresh_hash_selected_rank_only_shadow_population"
            ]
            and not source["authorization"][
                "load_model_before_population_prompt_controls_gates_and_invariance_lock"
            ]
            and file_sha256(PROJECT_ROOT / source["inventory"])
            == source["inventory_sha256"]
        ),
        "V90_model_free_authority_and_low_cost_shadow_decision_are_preserved": bool(
            payload_hash(parent_model_payload) == parent_model["lock_payload_sha256"]
            and parent_model["outcome"]["decision"]
            == "retain_model_free_authoritative_boundary"
            and parent_model["authorization"][
                "retain_qwen35_4b_only_as_frozen_shadow_baseline"
            ]
            and not parent_model["authorization"]["adopt_any_27b_or_8bit_condition"]
            and not parent_model["authorization"][
                "construct_small_large_union_or_cascade"
            ]
        ),
        "frozen_V79_exact_planner_evidence_is_exact": bool(
            payload_hash(planner_payload) == planner["lock_payload_sha256"]
            and planner_audit["passed"]
            and planner["decision"]
            == "freeze_V79_and_authorize_frozen_local_model_protocol_preregistration_only"
            and not planner["authorization"]["modify_or_rerun_V79"]
        ),
        "fresh_population_is_feasible_balanced_and_dialogue_unique": bool(
            len(selected) == config["population"]["recordCount"] == 64
            and strata == requested
            and len({row["dialogue_id"] for row in selected}) == 64
            and sum(row["active_intent"] == "NONE" for row in selected) == 32
            and sum(row["active_intent"] != "NONE" for row in selected) == 32
            and len({row["service"] for row in selected}) == 3
            and len(
                {
                    row["active_intent"]
                    for row in selected
                    if row["active_intent"] != "NONE"
                }
            )
            == 5
        ),
        "only_the_V90_retained_4B_shadow_snapshot_is_registered": bool(
            model["id"] == "qwen35_4b_4bit_rank_only"
            and model["repository"] == "mlx-community/Qwen3.5-4B-4bit"
            and model["revision"]
            == "0e7ffd5c629ef7719d4cbc04069232580bfa9d9c"
            and model["quantizationBits"] == 4
            and model["weightBytes"] == 3034300695
        ),
        "registered_manifest_is_exact_without_new_download_or_model_access": bool(
            file_sha256(PROJECT_ROOT / model["reuseManifest"])
            == model["reuseManifestFileSha256"]
            and not model["newWeightDownloadAuthorized"]
            and all(
                config["preDesignExposure"][key] == 0
                for key in (
                    "selectedUtteranceExtractionCount",
                    "manualUtteranceInspectionCount",
                    "newModelWeightDownloadCount",
                    "modelLoadCount",
                    "modelGenerationCount",
                    "LLMAPICallCount",
                )
            )
        ),
        "rank_only_contract_cannot_prune_mutate_state_update_belief_act_or_execute": bool(
            config["outputContract"]["completeSetAlwaysEqualsDeterministicSchemaSet"]
            and config["outputContract"]["noneAlwaysRetained"]
            and not config["outputContract"]["stateMutationPossible"]
            and not config["outputContract"]["posteriorOrActionAuthority"]
            and config["outputContract"]["permanentlyNonDeployable"]
            and not config["outputContract"]["executable"]
            and "append every omitted allowed identifier" in config["outputContract"][
                "canonicalCompletion"
            ]
            and config["decoding"]["temperature"] == 0.0
            and config["decoding"]["samplesPerRecord"] == 1
            and not config["decoding"]["enableThinking"]
            and not config["decoding"]["retryOnMalformedOutput"]
        ),
        "deterministic_and_oracle_ranking_controls_are_frozen": bool(
            set(config["rankingControls"])
            == {
                "schemaOrder",
                "lexicalOverlap",
                "identifierExactMatchGrammar",
                "exhaustiveUnordered",
                "oracleFirst",
            }
        ),
        "planner_invariance_is_exhaustive_and_model_independent": bool(
            config["plannerInvariance"]["requiredFixtureCount"] == 4
            and config["plannerInvariance"][
                "requiredHypothesisPermutationCountPerFixture"
            ]
            == 120
            and config["plannerInvariance"]["maximumAbsoluteValueError"] <= 1e-12
            and not config["plannerInvariance"]["modelOutputUsedByInvarianceHarness"]
        ),
        "utility_and_safety_gates_are_noncompensatory": bool(
            quality["minimumOverallTop1Rate"] >= 0.75
            and quality["minimumActiveTop1Rate"] >= 0.75
            and quality["minimumNoneTop1Rate"] >= 0.75
            and quality["minimumOverallTop2Rate"] >= 0.95
            and quality["minimumMRRImprovementOverBestNonOracleControl"] >= 0.05
            and quality["minimumMeanRankReductionVersusBestNonOracleControl"] >= 0.2
            and quality["minimumCanonicalCompleteSetRate"] == 1.0
            and quality["minimumCanonicalNoneRetentionRate"] == 1.0
            and quality["minimumAuthoritativeStatePreservationRate"] == 1.0
            and quality["minimumExactPlannerPermutationInvarianceRate"] == 1.0
            and quality["maximumPlannerActionMismatchCount"] == 0
            and quality["maximumExecutionCertificateViolationCount"] == 0
        ),
        "access_is_one_local_load_one_generation_per_record_and_zero_expansive_use": bool(
            access["requiredRecordCount"] == 64
            and access["maximumNewModelWeightDownloadCount"] == 0
            and access["maximumModelLoadCount"] == 1
            and access["maximumModelGenerationCount"] == 64
            and all(
                access[key] == 0
                for key in (
                    "maximumLLMAPICallCount",
                    "maximumAdapterTrainingRunCount",
                    "maximumManualUtteranceInspectionCount",
                    "maximumRealServiceCallCount",
                    "maximumExternalSideEffectCount",
                )
            )
        ),
        "pass_cannot_authorize_pruning_likelihood_belief_action_API_training_or_execution": bool(
            not config["decisionRule"]["passAuthorizesPruningOrEarlyStopping"]
            and not config["decisionRule"][
                "passAuthorizesLearnedLikelihoodOrBeliefAuthority"
            ]
            and not config["decisionRule"]["passAuthorizesActionSelectionOrExecution"]
            and not config["decisionRule"]["passAuthorizesAPIAccessOrAdapterTraining"]
        ),
        "design_stage_has_no_language_model_pruning_authority_or_side_effect": bool(
            stage["auditAndFreezeDesign"]
            and not stage["selectExtractAndSealCorpus"]
            and not stage["inspectUtterancesManually"]
            and not stage["loadOrRunLocalModel"]
            and not stage["runAPIModel"]
            and not stage["trainAdapter"]
            and not stage["pruneOrEarlyStopSearch"]
            and not stage["grantBeliefActionOrExecutionAuthority"]
            and not stage["performRealServiceCall"]
            and not stage["performExternalSideEffect"]
        ),
        "frozen_plan_protocol_tests_auditor_and_builder_exist": all(
            path.is_file()
            for path in (
                plan_path,
                protocol_path,
                tests_path,
                auditor_path,
                builder_path,
            )
        ),
    }
    passed = all(checks.values())
    audit = {
        "schema_version": "91-rank-only-design-audit",
        "experiment": "v91_rank_only_design_audit",
        "passed": passed,
        "decision": (
            "freeze_design_and_authorize_one_fresh_rank_only_corpus_seal"
            if passed
            else "reject_V91_rank_only_design"
        ),
        "checks": checks,
        "selected_structural_summary": {
            "record_count": len(selected),
            "dialogue_count": len({row["dialogue_id"] for row in selected}),
            "strata": {
                f"{key[0]}::{key[1]}": value
                for key, value in sorted(strata.items())
            },
            "record_id_sha256": payload_hash(
                {"ids": [row["record_id"] for row in selected]}
            ),
        },
        "access": {
            "selected_utterance_extraction_count": 0,
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
        "schema_version": "91-rank-only-design-lock",
        "experiment": "v91_rank_only_design_lock",
        "design": str(design_path.relative_to(PROJECT_ROOT)),
        "design_sha256": file_sha256(design_path),
        "config_payload": config,
        "source_outcome_lock": str(source_path.relative_to(PROJECT_ROOT)),
        "source_outcome_lock_sha256": file_sha256(source_path),
        "source_inventory": source["inventory"],
        "source_inventory_sha256": source["inventory_sha256"],
        "parent_model_decision_lock": str(parent_model_path.relative_to(PROJECT_ROOT)),
        "parent_model_decision_lock_sha256": file_sha256(parent_model_path),
        "planner_outcome_lock": str(planner_path.relative_to(PROJECT_ROOT)),
        "planner_outcome_lock_sha256": file_sha256(planner_path),
        "plan": str(plan_path.relative_to(PROJECT_ROOT)),
        "plan_sha256": file_sha256(plan_path),
        "protocol": str(protocol_path.relative_to(PROJECT_ROOT)),
        "protocol_sha256": file_sha256(protocol_path),
        "tests": str(tests_path.relative_to(PROJECT_ROOT)),
        "tests_sha256": file_sha256(tests_path),
        "auditor": str(auditor_path.relative_to(PROJECT_ROOT)),
        "auditor_sha256": file_sha256(auditor_path),
        "builder": str(builder_path.relative_to(PROJECT_ROOT)),
        "builder_sha256": file_sha256(builder_path),
        "design_audit": str(audit_path.relative_to(PROJECT_ROOT)),
        "design_audit_sha256": file_sha256(audit_path),
        "authorization": {
            "modify_population_model_prompt_decoding_controls_gates_or_decisions": False,
            "select_extract_and_seal_corpus_once": True,
            "manually_inspect_source_language": False,
            "implement_and_audit_ranker_runner_and_invariance_after_corpus_seal": True,
            "load_or_run_local_model_before_implementation_lock": False,
            "run_API_model_or_train_adapter": False,
            "prune_or_early_stop_search": False,
            "grant_model_belief_action_or_execution_authority": False,
            "perform_real_service_call_or_external_side_effect": False
        },
    }
    lock["lock_payload_sha256"] = payload_hash(lock)
    lock_path.write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n")
    print(json.dumps(audit, indent=2, sort_keys=True))
    print(
        json.dumps(
            {
                "lock": str(lock_path.relative_to(PROJECT_ROOT)),
                "sha256": file_sha256(lock_path),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
