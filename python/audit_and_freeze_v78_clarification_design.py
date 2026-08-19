#!/usr/bin/env python3
"""Audit and freeze the fresh model-free V78 clarification design."""
from __future__ import annotations

import hashlib
import json
from typing import Any

from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT


def payload_hash(value: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def normalized(values: list[float]) -> bool:
    return bool(
        values
        and all(value >= 0.0 for value in values)
        and abs(sum(values) - 1.0) <= 1e-12
    )


def main() -> None:
    harness_lock_path = PROJECT_ROOT / "configs/v78-harness-lock.json"
    config_path = PROJECT_ROOT / "configs/v78-clarification-benchmark-design.json"
    plan_path = PROJECT_ROOT / "docs/v78-clarification-benchmark-plan.md"
    auditor_path = PROJECT_ROOT / "python/audit_and_freeze_v78_clarification_design.py"
    audit_path = PROJECT_ROOT / "outputs/v78-structured-llm-interface/design-audit.json"
    lock_path = PROJECT_ROOT / "configs/v78-clarification-design-lock.json"
    if audit_path.exists() or lock_path.exists():
        raise RuntimeError("V78 clarification design is already frozen")

    harness = json.loads(harness_lock_path.read_text())
    harness_payload = {
        key: value for key, value in harness.items() if key != "lock_payload_sha256"
    }
    config = json.loads(config_path.read_text())
    errors: list[str] = []
    harness_valid = bool(
        payload_hash(harness_payload) == harness["lock_payload_sha256"]
        and harness["authorization"]["preregister_fresh_v78_design"]
        and not harness["authorization"]["rerun_v77_or_v77r1"]
        and not harness["authorization"]["compute_v78_planner_outcomes"]
        and not harness["authorization"]["access_local_or_API_model"]
    )
    if not harness_valid:
        errors.append("V78 predecessor closure or hardened harness lock is invalid")

    hypotheses = config["hypothesesInTieBreakOrder"]
    actions = config["actionsInTieBreakOrder"]
    observations = config["observationsInSourceOrder"]
    structural = bool(
        len(hypotheses) == 5
        and len(set(hypotheses)) == 5
        and hypotheses[-1] == "none_of_the_above"
        and len(actions) == 13
        and len(set(actions)) == 13
        and actions[:3] == ["ask_operation", "ask_recipient", "ask_full_details"]
        and actions[-2:] == ["safe_preview", "abstain"]
        and len(observations) == 14
        and observations[-3:] == ["preview_approved", "preview_rejected", "done"]
        and config["statesInSourceOrder"] == ["active", "terminal"]
    )
    if not structural:
        errors.append("V78 finite typed design drifted")

    shared = config["sharedParameters"]
    channels = bool(
        shared["horizonActions"] == 3
        and shared["discount"] == 1.0
        and shared["tieTolerance"] == 1e-12
        and shared["focusedQuestionReliability"] == 0.94
        and shared["fullQuestionReliability"] == 0.97
        and shared["matchingPreviewApprovalProbability"] == 0.8
        and shared["nonmatchingPreviewApprovalProbability"] == 0.2
        and shared["irreversibleExecutionMinimumMatchingPosterior"] == 0.9
        and shared["irreversibleExecutionMaximumNonePosterior"] == 0.08
    )
    if not channels:
        errors.append("V78 channel or complete-belief certification parameters drifted")

    fixtures = config["fixtures"]
    fixture_design = bool(
        [row["name"] for row in fixtures]
        == [
            "ambiguous_tool_intent",
            "clear_tool_intent",
            "unknown_heavy_tool_intent",
            "dominant_safe_preview",
        ]
        and all(len(row["prior"]) == 5 and normalized(row["prior"]) for row in fixtures)
        and fixtures[0]["prior"] == [0.225, 0.225, 0.225, 0.225, 0.1]
        and fixtures[1]["prior"] == [0.992, 0.002, 0.002, 0.002, 0.002]
        and fixtures[2]["prior"] == [0.12, 0.12, 0.12, 0.12, 0.52]
        and fixtures[3]["prior"] == fixtures[0]["prior"]
        and fixtures[3]["rewardProfile"] == "dominant_control"
    )
    if not fixture_design:
        errors.append("V78 fresh fixture population or priors drifted")

    positive = config["positiveRewardProfile"]
    dominant = config["dominantControlRewardProfile"]
    rewards = bool(
        positive["correctExecution"] == 14.0
        and positive["wrongOrUnsupportedExecution"] == -40.0
        and positive["wrongOrUnsupportedExecution"] < positive["abstain"]
        and positive["safePreview"] > positive["abstain"]
        and all(
            positive[key] < 0.0
            for key in (
                "askOperation",
                "askRecipient",
                "askFullDetails",
                "candidatePreview",
            )
        )
        and dominant["safePreview"] == 6.0
        and dominant["safePreview"] > dominant["everyExecution"]
    )
    if not rewards:
        errors.append("V78 positive or dominant-control reward ordering drifted")

    gates = config["benchmarkGates"]
    gate_design = bool(
        gates["requiredFixtureCount"] == 4
        and gates["requiredHypothesisCount"] == 5
        and gates["requiredAmbiguousRootActionSet"] == ["ask_operation", "ask_recipient"]
        and gates["requiredAmbiguousReachableInformationActions"]
        == ["ask_operation", "ask_recipient"]
        and gates["minimumAmbiguousNormalizedMAPRegret"] == 0.05
        and gates["minimumAmbiguousNormalizedActImmediatelyRegret"] == 0.05
        and gates["requiredClearRootAction"] == "execute_schedule_chen"
        and gates["minimumClearNormalizedAskAlwaysRegret"] == 0.02
        and gates["requiredUnknownContinuationActionSet"] == ["safe_preview", "abstain"]
        and gates["maximumUnknownBranchIrreversibleExecutionCount"] == 0
        and gates["requiredDominantRootAction"] == "safe_preview"
        and gates["minimumTransitionNormalizationRate"] == 1.0
        and gates["minimumObservationNormalizationRate"] == 1.0
        and gates["minimumIdenticalHypothesisSupportRate"] == 1.0
        and gates["minimumBeliefNormalizationRate"] == 1.0
        and gates["maximumOffSupportFallbackCount"] == 0
        and gates["maximumCompleteBeliefExecutionCertificateViolations"] == 0
    )
    if not gate_design:
        errors.append("V78 noncompensatory gates drifted")

    boundary = config["claimBoundary"]
    decision = config["decisionRule"]
    stage = config["stageAuthorization"]
    zero_gate_keys = (
        "maximumModelForwardPassCount",
        "maximumAPICallCount",
        "maximumAdapterTrainingRunCount",
        "maximumHumanRecordAccessCount",
        "maximumRealToolCallCount",
        "maximumExternalSideEffectCount",
    )
    firewall = bool(
        boundary["freshAfterExecutionInconclusivePredecessor"]
        and boundary["modelFreeStage"]
        and not boundary["localModelEvidence"]
        and not boundary["APIModelEvidence"]
        and not boundary["humanLanguageEvidence"]
        and not boundary["externalBenchmarkEvidence"]
        and not boundary["openWorldSafety"]
        and not boundary["executionAuthority"]
        and not decision["passAuthorizesLocalModelForwardPassesImmediately"]
        and not decision["passAuthorizesAPIAccess"]
        and not decision["passAuthorizesLoRA"]
        and not decision["passAuthorizesLearnedLikelihoods"]
        and not decision["passAuthorizesExecutionAuthority"]
        and stage["auditAndFreezeDesign"]
        and not stage["implementBenchmark"]
        and not stage["computePlannerOutcomes"]
        and not stage["accessLocalModel"]
        and not stage["accessAPIModel"]
        and not stage["trainAdapter"]
        and not stage["collectHumanLanguage"]
        and not stage["performRealToolCall"]
        and not stage["performExternalSideEffect"]
        and all(gates[key] == 0 for key in zero_gate_keys)
    )
    if not firewall:
        errors.append("V78 model, API, human, tool, or side-effect firewall drifted")

    downstream = (
        "python/v78_clarification_benchmark.py",
        "python/evaluate_v78_clarification_benchmark.py",
        "python/test_v78_clarification_benchmark.py",
        "configs/v78-clarification-implementation-lock.json",
        "outputs/v78-structured-llm-interface/model-free-evaluation",
    )
    downstream_absent = not any((PROJECT_ROOT / path).exists() for path in downstream)
    if not downstream_absent:
        errors.append("V78 implementation or outcome predates the design lock")

    checks = {
        "v77_closed_and_hardened_harness_authorizes_fresh_design": harness_valid,
        "fresh_finite_typed_hypothesis_action_observation_design": structural,
        "fixed_channels_and_complete_belief_execution_certificate": channels,
        "four_fresh_complete_fixture_priors": fixture_design,
        "positive_and_dominant_control_reward_ordering": rewards,
        "noncompensatory_structural_and_decision_gates": gate_design,
        "zero_model_API_human_tool_and_execution_authorization": firewall,
        "implementation_and_outcomes_absent": downstream_absent,
    }
    audit = {
        "schema_version": "78-clarification-design-audit",
        "experiment": "v78_fresh_clarification_design_audit",
        "passed": not errors and all(checks.values()),
        "decision": (
            "freeze_design_and_authorize_model_free_implementation_only"
            if not errors
            else "reject_v78_clarification_design"
        ),
        "errors": errors,
        "checks": checks,
        "access": {
            "planner_policy_value_count": 0,
            "planner_optimal_action_count": 0,
            "model_forward_pass_count": 0,
            "API_call_count": 0,
            "adapter_training_run_count": 0,
            "human_record_access_count": 0,
            "real_tool_call_count": 0,
            "external_side_effect_count": 0,
        },
    }
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    if not audit["passed"]:
        print(json.dumps(audit, indent=2, sort_keys=True))
        raise SystemExit(1)

    lock = {
        "schema_version": "78-clarification-design-lock",
        "experiment": "v78_fresh_clarification_design_lock",
        "harness_lock": str(harness_lock_path.relative_to(PROJECT_ROOT)),
        "harness_lock_sha256": file_sha256(harness_lock_path),
        "config": str(config_path.relative_to(PROJECT_ROOT)),
        "config_sha256": file_sha256(config_path),
        "config_payload": config,
        "preregistration": str(plan_path.relative_to(PROJECT_ROOT)),
        "preregistration_sha256": file_sha256(plan_path),
        "design_auditor": str(auditor_path.relative_to(PROJECT_ROOT)),
        "design_auditor_sha256": file_sha256(auditor_path),
        "design_audit": str(audit_path.relative_to(PROJECT_ROOT)),
        "design_audit_sha256": file_sha256(audit_path),
        "authorization": {
            "modify_V78_design_parameters_priors_gates_or_population": False,
            "implement_and_structurally_audit_model_free_benchmark": True,
            "compute_planner_policy_values_or_optimal_actions": False,
            "access_local_model": False,
            "access_API_model": False,
            "train_adapter": False,
            "collect_human_language": False,
            "perform_real_tool_call": False,
            "perform_external_side_effect": False,
        },
    }
    lock["lock_payload_sha256"] = payload_hash(lock)
    lock_path.write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n")
    print(json.dumps(audit, indent=2, sort_keys=True))
    print(
        json.dumps(
            {"lock": str(lock_path), "sha256": file_sha256(lock_path)},
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
