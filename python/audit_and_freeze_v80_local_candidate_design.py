#!/usr/bin/env python3
"""Audit and freeze V80 before corpus construction or local-model access."""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from typing import Any

from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT


def payload_hash(value: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def main() -> None:
    parent_path = PROJECT_ROOT / "configs/v79-terminal-utility-outcome-lock.json"
    config_path = PROJECT_ROOT / "configs/v80-local-candidate-generation-design.json"
    plan_path = PROJECT_ROOT / "docs/v80-local-candidate-generation-plan.md"
    auditor_path = PROJECT_ROOT / "python/audit_and_freeze_v80_local_candidate_design.py"
    audit_path = PROJECT_ROOT / "outputs/v80-local-candidate-generation/design-audit.json"
    lock_path = PROJECT_ROOT / "configs/v80-local-candidate-generation-design-lock.json"
    if audit_path.exists() or lock_path.exists():
        raise RuntimeError("V80 local candidate protocol is already frozen")

    parent = json.loads(parent_path.read_text())
    parent_payload = {
        key: value for key, value in parent.items() if key != "lock_payload_sha256"
    }
    config = json.loads(config_path.read_text())
    parent_valid = bool(
        payload_hash(parent_payload) == parent["lock_payload_sha256"]
        and parent["authorization"][
            "preregister_frozen_local_model_candidate_generation_protocol"
        ]
        and not parent["authorization"]["access_local_or_API_model"]
        and not parent["authorization"]["run_model_forward_pass_before_protocol_lock"]
    )

    model = config["model"]
    local_model = bool(
        model["provider"] == "local_mlx"
        and model["repository"] == "mlx-community/Qwen3.5-4B-4bit"
        and model["revision"] == "0e7ffd5c629ef7719d4cbc04069232580bfa9d9c"
        and model["adapterPath"] is None
        and model["frozen"]
        and not model["APIRequired"]
    )
    candidates = config["candidateIdsInRequiredOrder"]
    records = config["records"]
    counts = Counter(row["stratum"] for row in records)
    population = bool(
        len(candidates) == 5
        and len(set(candidates)) == 5
        and candidates[-1] == "none_of_the_above"
        and len(records) == 24
        and len({row["id"] for row in records}) == 24
        and len({row["instruction"] for row in records}) == 24
        and counts
        == Counter(
            {
                "clear": 8,
                "recipient_ambiguous": 4,
                "operation_ambiguous": 4,
                "fully_ambiguous": 4,
                "out_of_ontology": 4,
            }
        )
        and all(
            row["goldCandidateIds"]
            == [candidate for candidate in candidates if candidate in row["goldCandidateIds"]]
            and "none_of_the_above" in row["goldCandidateIds"]
            and len(set(row["goldCandidateIds"])) == len(row["goldCandidateIds"])
            for row in records
        )
    )
    contract = config["outputContract"]
    decoding = config["decoding"]
    prompt_contract = bool(
        contract["exactTopLevelKeys"] == ["candidate_ids"]
        and contract["minimumCandidates"] == 1
        and contract["maximumCandidates"] == 5
        and not contract["duplicatesAllowed"]
        and not contract["unknownIdsAllowed"]
        and not contract["markdownFencesAllowed"]
        and not contract["explanatoryTextAllowed"]
        and not contract["confidenceOrProbabilityFieldsAllowed"]
        and not contract["actionOrToolFieldsAllowed"]
        and contract["noneOfTheAboveRequired"]
        and "not a decision maker" in config["systemPrompt"]
        and "Do not assign probabilities or confidence" in config["systemPrompt"]
        and not decoding["enableThinking"]
        and decoding["temperature"] == 0.0
        and decoding["maximumNewTokens"] == 128
        and decoding["maximumPromptTokens"] == 512
        and decoding["samplesPerRecord"] == 1
        and not decoding["retryOnMalformedOutput"]
    )
    gates = config["gates"]
    gate_design = bool(
        gates["requiredRecordCount"] == 24
        and gates["requiredStratumCounts"] == dict(counts)
        and gates["minimumExactJSONParseRate"] == 1.0
        and gates["minimumSchemaValidityRate"] == 1.0
        and gates["minimumNoneOfTheAboveInclusionRate"] == 1.0
        and gates["minimumMeanGoldCandidateRecall"] == 0.95
        and gates["minimumPerStratumMeanGoldCandidateRecall"] == 0.9
        and gates["minimumExactCandidateSetAccuracy"] == 0.75
        and gates["minimumClearExactCandidateSetAccuracy"] == 0.75
        and gates["minimumOutOfOntologyExactCandidateSetAccuracy"] == 0.75
        and gates["minimumCanonicalOrderRate"] == 1.0
        and gates["maximumMeanCandidateCount"] == 3.5
        and gates["maximumModelForwardPassCount"] == 24
    )
    stage = config["stageAuthorization"]
    decision = config["decisionRule"]
    firewall = bool(
        stage["auditAndFreezeProtocol"]
        and not stage["constructAndSealCorpus"]
        and not stage["implementAndAuditLocalRunner"]
        and not stage["runLocalModel"]
        and not stage["runAPIModel"]
        and not stage["trainAdapter"]
        and not stage["collectHumanLanguage"]
        and not stage["performRealToolCall"]
        and not stage["performExternalSideEffect"]
        and not decision["passAuthorizesModelToBeliefIntegrationImmediately"]
        and not decision["passAuthorizesAPIAccess"]
        and not decision["passAuthorizesAdapterTraining"]
        and not decision["passAuthorizesLearnedLikelihoods"]
        and not decision["passAuthorizesRealToolExecution"]
        and all(
            gates[key] == 0
            for key in (
                "maximumAPICallCount",
                "maximumAdapterTrainingRunCount",
                "maximumHumanRecordAccessCount",
                "maximumRealToolCallCount",
                "maximumExternalSideEffectCount",
            )
        )
    )
    downstream = (
        "data/v80-local-candidate-generation",
        "python/build_v80_local_candidate_corpus.py",
        "python/run_v80_local_candidate_generation_mlx.py",
        "python/evaluate_v80_local_candidate_generation.py",
        "configs/v80-local-candidate-generation-implementation-lock.json",
        "outputs/v80-local-candidate-generation/evaluation",
    )
    downstream_absent = not any((PROJECT_ROOT / path).exists() for path in downstream)
    checks = {
        "V79_positive_model_free_outcome_authorizes_protocol_only": parent_valid,
        "frozen_revision_pinned_local_model_without_adapter_or_API": local_model,
        "complete_unique_canonical_24_record_population": population,
        "strict_candidate_only_output_and_deterministic_decoding": prompt_contract,
        "noncompensatory_candidate_generation_gates": gate_design,
        "zero_prelock_model_API_human_tool_or_training_authorization": firewall,
        "corpus_runner_and_outcome_absent": downstream_absent,
    }
    passed = all(checks.values())
    audit = {
        "schema_version": "80-local-candidate-generation-design-audit",
        "experiment": "v80_local_candidate_generation_design_audit",
        "passed": passed,
        "decision": (
            "freeze_protocol_and_authorize_corpus_runner_implementation_without_model_access"
            if passed
            else "reject_V80_protocol"
        ),
        "checks": checks,
        "stratum_counts": dict(sorted(counts.items())),
        "access": {
            "model_forward_pass_count": 0,
            "API_call_count": 0,
            "adapter_training_run_count": 0,
            "human_record_access_count": 0,
            "real_tool_call_count": 0,
            "external_side_effect_count": 0,
        },
    }
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    if not passed:
        print(json.dumps(audit, indent=2, sort_keys=True))
        raise SystemExit(1)

    lock = {
        "schema_version": "80-local-candidate-generation-design-lock",
        "experiment": "v80_local_candidate_generation_design_lock",
        "parent_outcome_lock": str(parent_path.relative_to(PROJECT_ROOT)),
        "parent_outcome_lock_sha256": file_sha256(parent_path),
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
            "modify_prompt_model_revision_population_decoding_or_gates": False,
            "construct_and_seal_corpus": True,
            "implement_and_audit_local_runner": True,
            "run_local_model": False,
            "run_API_model": False,
            "train_adapter": False,
            "collect_human_language": False,
            "perform_real_tool_call_or_external_side_effect": False,
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
