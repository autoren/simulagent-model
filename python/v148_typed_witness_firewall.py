from __future__ import annotations

from collections import Counter
from copy import deepcopy
from typing import Any


SUFFICIENT_KEYS = {"evidence_status", "operation", "object_class", "scope", "source"}
INSUFFICIENT_KEYS = {"evidence_status", "compatible_choice_ids", "source"}


def _signature(value: dict[str, Any]) -> tuple[str, str, str]:
    return value["operation"], value["object_class"], value["scope"]


def _base(config: dict[str, Any], llm_proposal: Any) -> dict[str, Any]:
    return {
        "llm_proposal_valid": isinstance(llm_proposal, str) and llm_proposal in config["outputIds"],
        "llm_proposal_non_authoritative": True,
        "authoritative_hypothesis_ids_retained": list(config["outputIds"]),
        "authoritative_hypothesis_universe_pruned": False,
        "capability_defined_or_registered": False,
        "executable": False,
        "actual_execution_count": 0,
    }


def _fail_closed(config: dict[str, Any], llm_proposal: Any, reason: str) -> dict[str, Any]:
    return {
        **_base(config, llm_proposal),
        "witness_valid": False,
        "validation_reason": reason,
        "normalized_witness": None,
        "final_state_id": config["insufficientId"],
        "final_output_structurally_valid": True,
    }


def finalize_witness(witness: Any, llm_proposal: Any, config: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(witness, dict):
        return _fail_closed(config, llm_proposal, "witness_not_object")
    status = witness.get("evidence_status")
    if status == "INSUFFICIENT":
        if set(witness) != INSUFFICIENT_KEYS:
            return _fail_closed(config, llm_proposal, "invalid_insufficient_shape")
        compatible = witness.get("compatible_choice_ids")
        if (
            witness.get("source") != config["trustedSource"]
            or not isinstance(compatible, list)
            or len(compatible) < 2
            or compatible != sorted(set(compatible))
            or any(choice not in config["outputIds"][:-1] for choice in compatible)
        ):
            return _fail_closed(config, llm_proposal, "invalid_insufficient_evidence")
        return {
            **_base(config, llm_proposal),
            "witness_valid": True,
            "validation_reason": "valid_insufficient_witness",
            "normalized_witness": deepcopy(witness),
            "final_state_id": config["insufficientId"],
            "final_output_structurally_valid": True,
        }
    if status != "SUFFICIENT":
        return _fail_closed(config, llm_proposal, "invalid_evidence_status")
    if set(witness) != SUFFICIENT_KEYS:
        return _fail_closed(config, llm_proposal, "invalid_sufficient_shape")
    if witness.get("source") != config["trustedSource"]:
        return _fail_closed(config, llm_proposal, "untrusted_witness_source")
    for field, allowed in config["allowedValues"].items():
        if not isinstance(witness.get(field), str) or witness[field] not in allowed:
            return _fail_closed(config, llm_proposal, f"invalid_{field}")

    signature = _signature(witness)
    known = {
        (row["operation"], row["object_class"], row["scope"]): row["choice_id"]
        for row in config["knownWitnesses"]
    }
    unsupported = {
        (row["operation"], row["object_class"], row["scope"])
        for row in config["unsupportedWitnesses"]
    }
    if signature in known:
        final_state = known[signature]
        reason = "exact_trusted_known_witness"
    elif signature in unsupported:
        final_state = config["unsupportedId"]
        reason = "exact_registered_forbidden_witness"
    else:
        final_state = config["novelCandidateId"]
        reason = "well_typed_unknown_mechanic_candidate"
    return {
        **_base(config, llm_proposal),
        "witness_valid": True,
        "validation_reason": reason,
        "normalized_witness": deepcopy(witness),
        "final_state_id": final_state,
        "final_output_structurally_valid": True,
    }


def build_valid_witnesses(config: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in config["knownWitnesses"]:
        rows.append({
            "case_id": f"known::{row['choice_id']}",
            "truth_state_id": row["choice_id"],
            "witness": {"evidence_status": "SUFFICIENT", "source": config["trustedSource"], **{key: row[key] for key in ("operation", "object_class", "scope")}},
        })
    for index, row in enumerate(config["novelWitnesses"]):
        rows.append({
            "case_id": f"novel::{index}",
            "truth_state_id": config["novelCandidateId"],
            "witness": {"evidence_status": "SUFFICIENT", "source": config["trustedSource"], **row},
        })
    for index, row in enumerate(config["unsupportedWitnesses"]):
        rows.append({
            "case_id": f"unsupported::{index}",
            "truth_state_id": config["unsupportedId"],
            "witness": {"evidence_status": "SUFFICIENT", "source": config["trustedSource"], **row},
        })
    for index, compatible in enumerate(config["insufficientCompatibleSets"]):
        rows.append({
            "case_id": f"insufficient::{index}",
            "truth_state_id": config["insufficientId"],
            "witness": {"evidence_status": "INSUFFICIENT", "compatible_choice_ids": sorted(compatible), "source": config["trustedSource"]},
        })
    return rows


def malformed_witnesses(config: dict[str, Any]) -> list[Any]:
    known = build_valid_witnesses(config)[0]["witness"]
    return [
        None,
        {},
        {key: value for key, value in known.items() if key != "scope"},
        {**known, "source": "MODEL_GENERATED"},
        {**known, "operation": "invent"},
        {"evidence_status": "INSUFFICIENT", "compatible_choice_ids": ["K31"], "source": config["trustedSource"]},
        {"evidence_status": "INSUFFICIENT", "compatible_choice_ids": ["K31", "K31"], "source": config["trustedSource"]},
        {**known, "extra": True},
        {**known, "scope": ["future"]},
        {"evidence_status": ["SUFFICIENT"], "source": config["trustedSource"]},
    ]


def near_known_mutations(config: dict[str, Any]) -> list[dict[str, Any]]:
    mutations = []
    for row in config["knownWitnesses"]:
        base = {"evidence_status": "SUFFICIENT", "source": config["trustedSource"], **{key: row[key] for key in ("operation", "object_class", "scope")}}
        for field in ("operation", "object_class", "scope"):
            replacement = next(value for value in config["allowedValues"][field] if value != base[field])
            mutations.append({"claimed_known_id": row["choice_id"], "witness": {**base, field: replacement}})
    return mutations


def evaluate(config: dict[str, Any]) -> dict[str, Any]:
    valid = build_valid_witnesses(config)
    oracle_rows = []
    by_case: dict[str, set[str]] = {}
    for case in valid:
        for proposal in config["outputIds"]:
            output = finalize_witness(case["witness"], proposal, config)
            oracle_rows.append({"case": case, "proposal": proposal, "output": output})
            by_case.setdefault(case["case_id"], set()).add(output["final_state_id"])
    malformed_rows = [finalize_witness(value, "K31", config) for value in malformed_witnesses(config)]
    near_rows = [
        {"mutation": row, "output": finalize_witness(row["witness"], row["claimed_known_id"], config)}
        for row in near_known_mutations(config)
    ]
    known_rows = [row for row in oracle_rows if row["case"]["truth_state_id"] in config["knownIds"]]
    accepted_known = [row for row in oracle_rows if row["output"]["final_state_id"] in config["knownIds"]]
    novel_rows = [row for row in oracle_rows if row["case"]["truth_state_id"] == config["novelCandidateId"]]
    unsupported_rows = [row for row in oracle_rows if row["case"]["truth_state_id"] == config["unsupportedId"]]
    insufficient_rows = [row for row in oracle_rows if row["case"]["truth_state_id"] == config["insufficientId"]]
    metrics = {
        "valid_witness_count": len(valid),
        "candidate_proposal_count": len(config["outputIds"]),
        "oracle_candidate_cross_product_count": len(oracle_rows),
        "malformed_witness_count": len(malformed_rows),
        "near_known_mutation_count": len(near_rows),
        "oracle_exact_accuracy": sum(row["output"]["final_state_id"] == row["case"]["truth_state_id"] for row in oracle_rows) / len(oracle_rows),
        "candidate_invariance": sum(len(outputs) == 1 for outputs in by_case.values()) / len(by_case),
        "known_acceptance_precision": sum(row["case"]["truth_state_id"] == row["output"]["final_state_id"] for row in accepted_known) / len(accepted_known),
        "known_acceptance_recall": sum(row["output"]["final_state_id"] == row["case"]["truth_state_id"] for row in known_rows) / len(known_rows),
        "novel_candidate_routing": sum(row["output"]["final_state_id"] == config["novelCandidateId"] for row in novel_rows) / len(novel_rows),
        "unsupported_routing": sum(row["output"]["final_state_id"] == config["unsupportedId"] for row in unsupported_rows) / len(unsupported_rows),
        "insufficient_routing": sum(row["output"]["final_state_id"] == config["insufficientId"] for row in insufficient_rows) / len(insufficient_rows),
        "malformed_fail_closed_rate": sum(not row["witness_valid"] and row["final_state_id"] == config["insufficientId"] for row in malformed_rows) / len(malformed_rows),
        "near_known_not_accepted_as_known_rate": sum(row["output"]["final_state_id"] not in config["knownIds"] for row in near_rows) / len(near_rows),
        "deterministic_final_output_validity": sum(row["output"]["final_output_structurally_valid"] for row in oracle_rows) / len(oracle_rows),
        "authoritative_true_hypothesis_retention": sum(row["output"]["authoritative_hypothesis_ids_retained"] == config["outputIds"] for row in oracle_rows) / len(oracle_rows),
        "output_state_counts": dict(sorted(Counter(row["output"]["final_state_id"] for row in oracle_rows).items())),
        "project_language_read_count": 0,
        "model_load_count": 0,
        "model_generation_or_score_count": 0,
        "API_call_count": 0,
        "training_run_count": 0,
        "actual_execution_count": 0,
    }
    gates = config["gates"]
    checks = {
        "population_counts": bool(
            metrics["valid_witness_count"] == config["population"]["validWitnessCount"]
            and metrics["candidate_proposal_count"] == config["population"]["candidateProposalCount"]
            and metrics["oracle_candidate_cross_product_count"] == config["population"]["oracleCandidateCrossProductCount"]
            and metrics["malformed_witness_count"] == config["population"]["malformedWitnessCount"]
            and metrics["near_known_mutation_count"] == config["population"]["nearKnownMutationCount"]
        ),
        "oracle_exact": metrics["oracle_exact_accuracy"] == gates["requiredOracleExactAccuracy"],
        "candidate_invariance": metrics["candidate_invariance"] == gates["requiredCandidateInvariance"],
        "known_precision": metrics["known_acceptance_precision"] == gates["requiredKnownAcceptancePrecision"],
        "known_recall": metrics["known_acceptance_recall"] == gates["requiredKnownAcceptanceRecall"],
        "novel_routing": metrics["novel_candidate_routing"] == gates["requiredNovelCandidateRouting"],
        "unsupported_routing": metrics["unsupported_routing"] == gates["requiredUnsupportedRouting"],
        "insufficient_routing": metrics["insufficient_routing"] == gates["requiredInsufficientRouting"],
        "malformed_fail_closed": metrics["malformed_fail_closed_rate"] == gates["requiredMalformedFailClosedRate"],
        "near_known_not_known": metrics["near_known_not_accepted_as_known_rate"] == gates["requiredNearKnownNotAcceptedAsKnownRate"],
        "final_output_validity": metrics["deterministic_final_output_validity"] == gates["requiredDeterministicFinalOutputValidity"],
        "authoritative_retention": metrics["authoritative_true_hypothesis_retention"] == gates["requiredAuthoritativeTrueHypothesisRetention"],
        "zero_language_model_API_training_execution": bool(
            metrics["project_language_read_count"] <= gates["maximumProjectLanguageReadCount"]
            and metrics["model_load_count"] <= gates["maximumModelLoadCount"]
            and metrics["model_generation_or_score_count"] <= gates["maximumModelGenerationOrScoreCount"]
            and metrics["API_call_count"] <= gates["maximumAPICallCount"]
            and metrics["training_run_count"] <= gates["maximumTrainingRunCount"]
            and metrics["actual_execution_count"] <= gates["maximumActualExecutionCount"]
        ),
    }
    return {"passed": all(checks.values()), "checks": checks, "metrics": metrics}


__all__ = ["build_valid_witnesses", "evaluate", "finalize_witness", "malformed_witnesses", "near_known_mutations"]
