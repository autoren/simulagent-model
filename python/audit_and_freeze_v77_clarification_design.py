#!/usr/bin/env python3
"""Audit and freeze the model-free V77 clarification benchmark design."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT


def payload_hash(value: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _normalized(probabilities: list[float]) -> bool:
    return bool(
        probabilities
        and all(value >= 0.0 for value in probabilities)
        and abs(sum(probabilities) - 1.0) <= 1e-12
    )


def main() -> None:
    config_path = PROJECT_ROOT / "configs/v77-clarification-benchmark-design.json"
    plan_path = PROJECT_ROOT / "docs/v77-clarification-benchmark-plan.md"
    v76_path = PROJECT_ROOT / "configs/v76-discovery-clean-source-census-lock.json"
    v58_path = PROJECT_ROOT / "docs/v58-deferred-status.md"
    auditor_path = PROJECT_ROOT / "python/audit_and_freeze_v77_clarification_design.py"
    audit_path = PROJECT_ROOT / "outputs/v77-structured-llm-interface/design-audit.json"
    lock_path = PROJECT_ROOT / "configs/v77-clarification-design-lock.json"
    if lock_path.exists():
        raise RuntimeError("V77 clarification design is already frozen")

    config = json.loads(config_path.read_text())
    v76 = json.loads(v76_path.read_text())
    v76_payload = {key: value for key, value in v76.items() if key != "lock_payload_sha256"}
    v58 = v58_path.read_text()
    errors: list[str] = []

    prior_program_closed = bool(
        payload_hash(v76_payload) == v76["lock_payload_sha256"]
        and v76["decision"]
        == "freeze_source_feasibility_deferral_before_any_candidate_implementation"
        and not v76["authorization"]["inspect_or_select_any_V76_candidate_implementation"]
        and not v76["authorization"]["compute_any_V76_candidate_policy_value_or_decision_statistic"]
        and "deferred without release or collection" in v58
        and "Model-generated utterances may be used in a separately named development stress test"
        in v58
    )
    if not prior_program_closed:
        errors.append("V76 active-sensing or V58 human-language closure is invalid")

    hypotheses = config["hypothesesInTieBreakOrder"]
    actions = config["actionsInTieBreakOrder"]
    observations = config["observationsInSourceOrder"]
    fixtures = config["fixtures"]
    structural_design = bool(
        len(hypotheses) == 5
        and hypotheses[-1] == "none_of_the_above"
        and len(set(hypotheses)) == len(hypotheses)
        and len(actions) == 13
        and len(set(actions)) == len(actions)
        and actions[:3] == ["ask_report", "ask_recipient", "ask_full_details"]
        and actions[-2:] == ["safe_draft", "abstain"]
        and len(observations) == 14
        and observations[-3:] == ["draft_approved", "draft_rejected", "done"]
        and config["statesInSourceOrder"] == ["active", "terminal"]
    )
    if not structural_design:
        errors.append("V77 hypothesis, action, observation, or state design drifted")

    shared = config["sharedParameters"]
    channel_and_certification = bool(
        shared["horizonActions"] == 3
        and shared["discount"] == 1.0
        and shared["tieTolerance"] == 1e-12
        and 0.5 < shared["focusedQuestionReliability"] < 1.0
        and 0.5 < shared["fullQuestionReliability"] < 1.0
        and shared["matchingDraftApprovalProbability"]
        > shared["nonmatchingDraftApprovalProbability"]
        and shared["irreversibleSendMinimumMatchingPosterior"] == 0.9
        and shared["irreversibleSendMaximumNonePosterior"] == 0.1
    )
    if not channel_and_certification:
        errors.append("V77 evidence channels or fail-closed send thresholds drifted")

    fixture_design = bool(
        len(fixtures) == 4
        and [row["name"] for row in fixtures]
        == [
            "ambiguous_send",
            "clear_send",
            "unknown_heavy",
            "dominant_safe_draft",
        ]
        and all(_normalized(row["prior"]) for row in fixtures)
        and all(len(row["prior"]) == len(hypotheses) for row in fixtures)
        and fixtures[0]["prior"][-1] > 0.0
        and fixtures[1]["prior"][0] == 0.995
        and fixtures[2]["prior"][-1] == 0.4
        and fixtures[3]["prior"] == fixtures[0]["prior"]
        and fixtures[3]["rewardProfile"] == "dominant_control"
    )
    if not fixture_design:
        errors.append("V77 fixture priors, membership, or negative control drifted")

    positive = config["positiveRewardProfile"]
    negative = config["dominantControlRewardProfile"]
    reward_design = bool(
        positive["correctSend"] > 0.0
        and positive["wrongOrUnsupportedSend"] < positive["abstain"]
        and positive["safeDraft"] > positive["abstain"]
        and positive["askReport"] < 0.0
        and positive["askRecipient"] < 0.0
        and positive["askFullDetails"] < 0.0
        and negative["safeDraft"] > negative["everySend"]
        and negative["safeDraft"] > 0.0
    )
    if not reward_design:
        errors.append("V77 positive or dominant-control reward ordering drifted")

    gates = config["benchmarkGates"]
    gate_design = bool(
        gates["requiredFixtureCount"] == 4
        and gates["requiredHypothesisCount"] == 5
        and gates["requiredNoneHypothesis"] == "none_of_the_above"
        and gates["requiredAmbiguousRootActionSet"]
        == ["ask_report", "ask_recipient"]
        and gates["requiredAmbiguousReachableInformationActions"]
        == ["ask_report", "ask_recipient"]
        and gates["minimumAmbiguousNormalizedMAPRegret"] == 0.05
        and gates["minimumAmbiguousNormalizedActImmediatelyRegret"] == 0.05
        and gates["requiredClearRootAction"] == "send_q2_lee"
        and gates["minimumClearNormalizedAskAlwaysRegret"] == 0.02
        and gates["requiredUnknownContinuationActionSet"]
        == ["safe_draft", "abstain"]
        and gates["requiredDominantRootAction"] == "safe_draft"
        and gates["minimumObservationNormalizationRate"] == 1.0
        and gates["minimumIdenticalHypothesisSupportRate"] == 1.0
        and gates["minimumBeliefNormalizationRate"] == 1.0
        and gates["maximumOffSupportFallbackCount"] == 0
    )
    if not gate_design:
        errors.append("V77 noncompensatory structural or decision gates drifted")

    boundary = config["claimBoundary"]
    decision = config["decisionRule"]
    stage = config["stageAuthorization"]
    firewall = bool(
        boundary["projectAuthoredDevelopmentMechanism"]
        and boundary["exactFiniteBenchmark"]
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
        and not stage["performExternalSideEffect"]
        and all(
            gates[key] == 0
            for key in (
                "maximumModelForwardPassCount",
                "maximumAPICallCount",
                "maximumAdapterTrainingRunCount",
                "maximumHumanRecordAccessCount",
                "maximumExternalSideEffectCount",
            )
        )
    )
    if not firewall:
        errors.append("V77 model, API, human-data, or execution firewall drifted")

    downstream_paths = (
        "python/v77_clarification_benchmark.py",
        "python/evaluate_v77_clarification_benchmark.py",
        "python/verify_v77_clarification_benchmark.py",
        "python/test_v77_clarification_benchmark.py",
        "configs/v77-clarification-implementation-lock.json",
        "configs/v77-clarification-outcome-lock.json",
        "outputs/v77-structured-llm-interface/model-free-evaluation",
        "data/v77-structured-llm-interface",
    )
    downstream_absent = not any((PROJECT_ROOT / path).exists() for path in downstream_paths)
    if not downstream_absent:
        errors.append("V77 implementation, outcomes, model data, or model access predates design lock")

    checks = {
        "V76_and_V58_boundaries_preserved": prior_program_closed,
        "finite_typed_hypothesis_action_observation_design": structural_design,
        "fixed_evidence_channels_and_fail_closed_certification": channel_and_certification,
        "four_complete_preregistered_fixture_priors": fixture_design,
        "positive_and_dominant_control_reward_ordering": reward_design,
        "noncompensatory_structural_and_decision_gates": gate_design,
        "zero_model_API_human_and_execution_authorization": firewall,
        "implementation_outcomes_and_model_artifacts_absent": downstream_absent,
    }
    audit = {
        "schema_version": "77-clarification-benchmark-design",
        "experiment": "v77_clarification_design_audit",
        "passed": not errors and all(checks.values()),
        "decision": (
            "freeze_design_and_authorize_model_free_implementation_only"
            if not errors
            else "reject_v77_clarification_design"
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
            "external_side_effect_count": 0,
        },
    }
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    if not audit["passed"]:
        print(json.dumps(audit, indent=2, sort_keys=True))
        raise SystemExit(1)

    lock = {
        "schema_version": "77-clarification-benchmark-design",
        "experiment": "v77_clarification_design_lock",
        "V76_outcome_lock": str(v76_path.relative_to(PROJECT_ROOT)),
        "V76_outcome_lock_sha256": file_sha256(v76_path),
        "V58_deferral": str(v58_path.relative_to(PROJECT_ROOT)),
        "V58_deferral_sha256": file_sha256(v58_path),
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
            "modify_V77_design_parameters_priors_or_gates": False,
            "implement_and_structurally_audit_model_free_benchmark": True,
            "compute_planner_policy_values_or_optimal_actions": False,
            "access_local_model": False,
            "access_API_model": False,
            "train_adapter": False,
            "collect_human_language": False,
            "perform_external_side_effect": False,
        },
    }
    lock["lock_payload_sha256"] = payload_hash(lock)
    lock_path.write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n")
    print(json.dumps(audit, indent=2, sort_keys=True))
    print(json.dumps({"lock": str(lock_path), "sha256": file_sha256(lock_path)}, indent=2))


if __name__ == "__main__":
    main()
