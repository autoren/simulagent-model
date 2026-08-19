from __future__ import annotations

import json
from typing import Any


def _sufficient_code(choice_id: str) -> str:
    return f"S__{choice_id}"


def _insufficient_code(pair: list[str]) -> str:
    return "I__" + "__".join(sorted(pair))


def build_codebook(config: dict[str, Any]) -> list[dict[str, Any]]:
    entries = []
    for choice_id in config["choiceIds"]:
        if choice_id == "A00":
            continue
        entries.append(
            {
                "certificate_code": _sufficient_code(choice_id),
                "certificate": {
                    "evidence_status": "SUFFICIENT",
                    "compatible_choice_ids": [choice_id],
                    "proposed_choice_id": choice_id,
                },
                "final_choice_id": choice_id,
            }
        )
    for pair in config["ambiguousPairs"]:
        sorted_pair = sorted(pair)
        entries.append(
            {
                "certificate_code": _insufficient_code(sorted_pair),
                "certificate": {
                    "evidence_status": "INSUFFICIENT",
                    "compatible_choice_ids": sorted_pair,
                    "proposed_choice_id": "A00",
                },
                "final_choice_id": "A00",
            }
        )
    return sorted(entries, key=lambda row: row["certificate_code"])


def finalize_code(code: Any, codebook: list[dict[str, Any]]) -> dict[str, Any]:
    by_code = {row["certificate_code"]: row for row in codebook}
    selected = by_code.get(code) if isinstance(code, str) else None
    final_choice = selected["final_choice_id"] if selected else "A00"
    return {
        "code_valid": selected is not None,
        "validation_reason": "valid_registered_code" if selected else "unknown_or_malformed_code",
        "normalized_certificate": selected["certificate"] if selected else None,
        "final_choice_id": final_choice,
        "final_json": json.dumps({"choice_id": final_choice}, sort_keys=True, separators=(",", ":")),
        "final_output_structurally_valid": True,
        "permanently_non_authoritative": True,
        "authoritative_hypothesis_universe_pruned": False,
        "actual_execution_count": 0,
    }


def build_abstract_population(config: dict[str, Any]) -> list[dict[str, Any]]:
    stages = config["abstractPopulation"]["stages"]
    rows = []
    for family_index, pair in enumerate(config["ambiguousPairs"]):
        left, right = pair
        for group_index in range(config["abstractPopulation"]["groupsPerFamily"]):
            group_id = f"v145-g{family_index:02d}-{group_index:02d}"
            specs = {
                "clear_known_familiar": (left, [left]),
                "clear_known_unfamiliar": (left, [left]),
                "clear_right": (right, [right]),
                "ambiguous": ("A00", sorted(pair)),
                "clarified_known": (left, [left]),
                "clarified_right": (right, [right]),
            }
            for stage in stages:
                truth, compatible = specs[stage]
                rows.append(
                    {
                        "fixture_id": f"{group_id}::{stage}",
                        "group_id": group_id,
                        "family_index": family_index,
                        "stage": stage,
                        "truth_choice_id": truth,
                        "compatible_choice_ids": compatible,
                    }
                )
    return rows


def oracle_code(row: dict[str, Any]) -> str:
    if row["truth_choice_id"] == "A00":
        return _insufficient_code(row["compatible_choice_ids"])
    return _sufficient_code(row["truth_choice_id"])


def evaluate(config: dict[str, Any]) -> dict[str, Any]:
    codebook = build_codebook(config)
    population = build_abstract_population(config)
    oracle_rows = []
    for row in population:
        code = oracle_code(row)
        finalized = finalize_code(code, codebook)
        certificate = finalized["normalized_certificate"]
        expected_status = "INSUFFICIENT" if row["truth_choice_id"] == "A00" else "SUFFICIENT"
        oracle_rows.append(
            {
                "code_valid": finalized["code_valid"],
                "certificate_exact": bool(
                    certificate
                    and certificate["evidence_status"] == expected_status
                    and certificate["compatible_choice_ids"] == row["compatible_choice_ids"]
                    and certificate["proposed_choice_id"] == row["truth_choice_id"]
                ),
                "final_exact": finalized["final_choice_id"] == row["truth_choice_id"],
                "final_output_valid": finalized["final_output_structurally_valid"],
            }
        )
    mutations: list[Any] = [None, "", "UNKNOWN", {}, [], 17, "s__K11", "S__A00", "I__K11", "I__K11__N99"]
    mutation_rows = [finalize_code(value, codebook) for value in mutations]
    wrong_code = _sufficient_code(config["ambiguousPairs"][0][0])
    wrong_truth = config["ambiguousPairs"][0][1]
    wrong = finalize_code(wrong_code, codebook)
    sufficient_count = sum(row["certificate"]["evidence_status"] == "SUFFICIENT" for row in codebook)
    insufficient_count = sum(row["certificate"]["evidence_status"] == "INSUFFICIENT" for row in codebook)
    unique_certificates = {
        json.dumps(row["certificate"], sort_keys=True, separators=(",", ":")) for row in codebook
    }
    metrics = {
        "code_count": len(codebook),
        "sufficient_code_count": sufficient_count,
        "insufficient_code_count": insufficient_count,
        "unique_certificate_count": len(unique_certificates),
        "abstract_fixture_count": len(population),
        "oracle_code_validity": sum(row["code_valid"] for row in oracle_rows) / len(oracle_rows),
        "oracle_certificate_exact_accuracy": sum(row["certificate_exact"] for row in oracle_rows) / len(oracle_rows),
        "oracle_final_choice_exact_accuracy": sum(row["final_exact"] for row in oracle_rows) / len(oracle_rows),
        "invalid_code_fail_closed_rate": sum(
            not row["code_valid"] and row["final_choice_id"] == "A00" and row["final_output_structurally_valid"]
            for row in mutation_rows
        ) / len(mutation_rows),
        "deterministic_final_output_validity": (
            sum(row["final_output_valid"] for row in oracle_rows)
            + sum(row["final_output_structurally_valid"] for row in mutation_rows)
        ) / (len(oracle_rows) + len(mutation_rows)),
        "registered_wrong_singleton_is_structurally_valid_but_semantically_wrong": bool(
            wrong["code_valid"] and wrong["final_choice_id"] != wrong_truth
        ),
        "authoritative_true_hypothesis_retention": 1.0,
        "project_language_read_count": 0,
        "model_load_count": 0,
        "model_generation_count": 0,
        "API_call_count": 0,
        "training_run_count": 0,
        "actual_execution_count": 0,
    }
    gates = config["gates"]
    checks = {
        "code_count": metrics["code_count"] == gates["requiredCodeCount"],
        "sufficient_code_count": metrics["sufficient_code_count"] == gates["requiredSufficientCodeCount"],
        "insufficient_code_count": metrics["insufficient_code_count"] == gates["requiredInsufficientCodeCount"],
        "unique_certificates": metrics["unique_certificate_count"] == gates["requiredUniqueCertificateCount"],
        "abstract_fixture_count": metrics["abstract_fixture_count"] == gates["requiredAbstractFixtureCount"],
        "oracle_code_validity": metrics["oracle_code_validity"] == gates["requiredOracleCodeValidity"],
        "oracle_certificate_exact": metrics["oracle_certificate_exact_accuracy"] == gates["requiredOracleCertificateExactAccuracy"],
        "oracle_final_exact": metrics["oracle_final_choice_exact_accuracy"] == gates["requiredOracleFinalChoiceExactAccuracy"],
        "invalid_codes_fail_closed": metrics["invalid_code_fail_closed_rate"] == gates["requiredInvalidCodeFailClosedRate"],
        "deterministic_final_output_validity": metrics["deterministic_final_output_validity"] == gates["requiredDeterministicFinalOutputValidity"],
        "wrong_singleton_limitation": metrics["registered_wrong_singleton_is_structurally_valid_but_semantically_wrong"] == gates["requiredRegisteredWrongSingletonSemanticLimitation"],
        "authoritative_retention": metrics["authoritative_true_hypothesis_retention"] == gates["requiredAuthoritativeTrueHypothesisRetention"],
        "zero_language_model_API_training_execution": all(
            metrics[key] == 0
            for key in ("project_language_read_count", "model_load_count", "model_generation_count", "API_call_count", "training_run_count", "actual_execution_count")
        ),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "metrics": metrics,
        "codebook": codebook,
        "mutation_count": len(mutations),
        "semantic_limitation": {
            "registered_wrong_code": wrong_code,
            "hidden_truth": wrong_truth,
            "structurally_valid": wrong["code_valid"],
            "structural_interface_cannot_detect_semantic_error": True,
        },
    }


__all__ = ["build_abstract_population", "build_codebook", "evaluate", "finalize_code", "oracle_code"]
