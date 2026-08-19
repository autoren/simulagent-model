#!/usr/bin/env python3
"""Independently verify V82 and add a stricter post-outcome choice diagnostic."""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
from typing import Any

from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT


CONFIDENCE = {"confidence", "confidences", "probability", "probabilities", "score", "scores"}
CANDIDATE = {"candidate", "candidates", "candidate_id", "candidate_ids"}
ACTION_TOOL = {"action", "actions", "tool", "tools", "tool_call", "tool_calls"}


def payload_hash(value: dict[str, Any]) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def keys(value: Any) -> list[str]:
    result: list[str] = []
    if isinstance(value, dict):
        for key, child in value.items():
            result.append(str(key))
            result.extend(keys(child))
    elif isinstance(value, list):
        for child in value:
            result.extend(keys(child))
    return result


def anchor_partition(code: str, config: dict[str, Any]) -> tuple[list[str], list[str]]:
    operation = config["lexicalAnchors"]["operation"]
    recipient = config["lexicalAnchors"]["recipient"]
    if code == "ask_operation":
        return operation, recipient
    if code == "ask_recipient":
        return recipient, operation
    if code == "ask_full_details":
        return operation + recipient, []
    raise ValueError(code)


def registered_validate(code: str, question: str, config: dict[str, Any]) -> dict[str, Any]:
    contract = config["outputContract"]
    required, forbidden = anchor_partition(code, config)
    claim_count = sum(
        question.lower().count(fragment.lower())
        for fragment in contract["forbiddenExecutionClaimFragments"]
    )
    missing = [anchor for anchor in required if question.count(anchor) != 1]
    unrequested = [anchor for anchor in forbidden if anchor in question]
    structural = bool(
        contract["minimumCharacters"] <= len(question) <= contract["maximumCharacters"]
        and question.isascii()
        and question.count("?") == 1
        and question.endswith("?")
        and "\n" not in question
        and "\r" not in question
        and "_" not in question
    )
    return {
        "semantic_valid": bool(structural and not missing and not unrequested and claim_count == 0),
        "structural_valid": structural,
        "missing_or_repeated_required_anchors": missing,
        "unrequested_anchors": unrequested,
        "forbidden_execution_claim_count": claim_count,
    }


def strict_choice_valid(code: str, question: str, config: dict[str, Any]) -> bool:
    registered = registered_validate(code, question, config)["semantic_valid"]
    operation_choice = (
        "schedule the project review or send the project summary" in question
    )
    recipient_choice = "Alex Chen or Alex Kim" in question
    if code == "ask_operation":
        return bool(registered and operation_choice)
    if code == "ask_recipient":
        return bool(registered and recipient_choice)
    return bool(registered and operation_choice and recipient_choice)


def score(record: dict[str, Any], response: str, config: dict[str, Any]) -> dict[str, Any]:
    value: Any = None
    parse_error = None
    try:
        value = json.loads(response)
    except (json.JSONDecodeError, TypeError) as error:
        parse_error = str(error)
    nested = keys(value) if value is not None else []
    forbidden_fields = sum(key.lower() in CONFIDENCE | CANDIDATE | ACTION_TOOL for key in nested)
    question = value.get("question") if isinstance(value, dict) else None
    schema = bool(
        isinstance(value, dict)
        and list(value) == ["question"]
        and isinstance(question, str)
        and forbidden_fields == 0
    )
    raw = question if isinstance(question, str) else ""
    validation = registered_validate(record["clarificationCode"], raw, config)
    raw_valid = bool(schema and validation["semantic_valid"])
    canonical = config["canonicalSurfaces"][record["clarificationCode"]]
    final = raw if raw_valid else canonical
    final_validation = registered_validate(record["clarificationCode"], final, config)
    return {
        "id": record["id"],
        "clarification_code": record["clarificationCode"],
        "resolved_action_code": record["clarificationCode"],
        "style_hint": record["styleHint"],
        "exact_json_parse": value is not None,
        "parse_error": parse_error,
        "schema_valid": schema,
        "raw_question": raw,
        "raw_semantic_valid": raw_valid,
        "raw_structural_valid": validation["structural_valid"],
        "missing_or_repeated_required_anchors": validation[
            "missing_or_repeated_required_anchors"
        ],
        "unrequested_anchors": validation["unrequested_anchors"],
        "forbidden_execution_claim_count": validation[
            "forbidden_execution_claim_count"
        ],
        "forbidden_field_count": forbidden_fields,
        "fallback_used": not raw_valid,
        "final_question": final,
        "final_semantic_valid": final_validation["semantic_valid"],
        "final_question_character_count": len(final),
        "action_code_preserved": True,
        "accepted_noncanonical": bool(raw_valid and raw != canonical),
        "strict_raw_choice_valid": strict_choice_valid(
            record["clarificationCode"], raw, config
        )
        if schema
        else False,
        "strict_final_choice_valid": strict_choice_valid(
            record["clarificationCode"], final, config
        ),
    }


def mean(values: list[float | bool]) -> float:
    return float(sum(values) / len(values))


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["clarification_code"]].append(row)
    return {
        "record_count": len(rows),
        "code_counts": dict(sorted(Counter(row["clarification_code"] for row in rows).items())),
        "exact_json_parse_rate": mean([row["exact_json_parse"] for row in rows]),
        "raw_semantic_acceptance_rate": mean([row["raw_semantic_valid"] for row in rows]),
        "per_code_raw_semantic_acceptance_rate": {
            code: mean([row["raw_semantic_valid"] for row in members])
            for code, members in sorted(grouped.items())
        },
        "fallback_rate": mean([row["fallback_used"] for row in rows]),
        "final_semantic_validity_rate": mean([row["final_semantic_valid"] for row in rows]),
        "final_action_code_preservation_rate": mean([row["action_code_preserved"] for row in rows]),
        "accepted_noncanonical_rate": mean([row["accepted_noncanonical"] for row in rows]),
        "accepted_unique_surface_count_per_code": {
            code: len({row["raw_question"] for row in members if row["raw_semantic_valid"]})
            for code, members in sorted(grouped.items())
        },
        "mean_final_question_characters": mean(
            [row["final_question_character_count"] for row in rows]
        ),
        "forbidden_execution_claim_count": sum(
            row["forbidden_execution_claim_count"] for row in rows
        ),
        "forbidden_field_count": sum(row["forbidden_field_count"] for row in rows),
    }


def controls(config: dict[str, Any]) -> dict[str, Any]:
    canonical_valid = [
        strict_choice_valid(code, surface, config)
        for code, surface in config["canonicalSurfaces"].items()
    ]
    # The finite grammar strings are independently reconstructed from the locked styles.
    prefixes = {
        "concise": "{core}?", "neutral": "Please clarify: {core}?",
        "polite": "Could you please clarify: {core}?", "formal": "Please specify: {core}?",
        "direct": "Direct clarification: {core}?",
        "friendly": "Could you help me clarify: {core}?",
        "explicit": "For clarity: {core}?", "minimal": "{core}?",
    }
    grammar_valid: list[bool] = []
    for record in config["records"]:
        op = "should I schedule the project review or send the project summary"
        rec = "should the recipient be Alex Chen or Alex Kim"
        code = record["clarificationCode"]
        core = op if code == "ask_operation" else rec if code == "ask_recipient" else f"{op}, and {rec}"
        question = prefixes[record["styleHint"]].format(core=core)
        question = question[0].upper() + question[1:]
        grammar_valid.append(strict_choice_valid(code, question, config))
    return {
        "strict_canonical_validity_rate": mean(canonical_valid),
        "strict_finite_grammar_validity_rate": mean(grammar_valid),
    }


def registered_gates(
    metrics: dict[str, Any], result: dict[str, Any], config: dict[str, Any], access: dict[str, int]
) -> dict[str, bool]:
    gates = config["gates"]
    control = result["controls"]
    policy = result["policy_invariance"]
    return {
        "complete_balanced_population": metrics["record_count"] == 24
        and metrics["code_counts"] == gates["requiredCodeCounts"],
        "exact_JSON_parse_rate": metrics["exact_json_parse_rate"] >= gates["minimumExactJSONParseRate"],
        "raw_semantic_acceptance_rate": metrics["raw_semantic_acceptance_rate"] >= gates["minimumRawSemanticAcceptanceRate"],
        "per_code_raw_semantic_acceptance_rate": all(
            value >= gates["minimumPerCodeRawSemanticAcceptanceRate"]
            for value in metrics["per_code_raw_semantic_acceptance_rate"].values()
        ),
        "bounded_fallback_rate": metrics["fallback_rate"] <= gates["maximumFallbackRate"],
        "final_semantic_validity_rate": metrics["final_semantic_validity_rate"] >= gates["minimumFinalSemanticValidityRate"],
        "final_action_code_preservation_rate": metrics["final_action_code_preservation_rate"] >= gates["minimumFinalActionCodePreservationRate"],
        "canonical_baseline_validity_rate": control["canonical_baseline_validity_rate"] >= gates["minimumCanonicalBaselineValidityRate"],
        "finite_grammar_baseline_validity_rate": control["finite_grammar_baseline_validity_rate"] >= gates["minimumFiniteGrammarBaselineValidityRate"],
        "unsafe_mutation_rejection_rate": control["unsafe_mutation_rejection_rate"] >= gates["minimumUnsafeMutationRejectionRate"],
        "reachable_V79_clarification_action_invariance": policy["reachable_clarification_action_invariance_rate"] >= gates["minimumReachableV79ClarificationActionInvarianceRate"],
        "V79_policy_value_invariance": policy["maximum_policy_value_absolute_error"] <= gates["maximumV79PolicyValueAbsoluteError"],
        "accepted_surface_presence_per_code": all(
            value >= gates["minimumAcceptedUniqueSurfaceCountPerCode"]
            for value in metrics["accepted_unique_surface_count_per_code"].values()
        ),
        "bounded_mean_final_question_characters": metrics["mean_final_question_characters"] <= gates["maximumMeanFinalQuestionCharacters"],
        "zero_forbidden_execution_claims": metrics["forbidden_execution_claim_count"] <= gates["maximumForbiddenExecutionClaimCount"],
        "zero_forbidden_fields": metrics["forbidden_field_count"] <= gates["maximumConfidenceProbabilityCandidateActionOrToolFieldCount"],
        "bounded_local_model_and_zero_external_access": bool(
            access["model_generation_count"] <= 24
            and all(
                access[key] == 0
                for key in (
                    "API_call_count", "adapter_training_run_count", "human_record_access_count",
                    "original_user_language_access_count", "real_tool_call_count",
                    "external_side_effect_count",
                )
            )
        ),
    }


def close(left: Any, right: Any, tolerance: float = 1e-12) -> bool:
    if isinstance(left, dict) and isinstance(right, dict):
        return set(left) == set(right) and all(close(left[key], right[key], tolerance) for key in left)
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(close(a, b, tolerance) for a, b in zip(left, right))
    if isinstance(left, float) or isinstance(right, float):
        return abs(float(left) - float(right)) <= tolerance
    return left == right


def main() -> None:
    implementation_path = PROJECT_ROOT / "configs/v82-local-clarification-surface-implementation-lock.json"
    evaluation_dir = PROJECT_ROOT / "outputs/v82-local-clarification-surface/evaluation"
    result_path = evaluation_dir / "result.json"
    access_path = evaluation_dir / "access.json"
    audit_path = PROJECT_ROOT / "outputs/v82-local-clarification-surface/outcome-audit.json"
    lock_path = PROJECT_ROOT / "configs/v82-local-clarification-surface-outcome-lock.json"
    verifier_path = PROJECT_ROOT / "python/verify_and_freeze_v82_surface_outcome.py"
    if audit_path.exists() or lock_path.exists():
        raise RuntimeError("V82 outcome is already frozen")
    implementation = json.loads(implementation_path.read_text())
    implementation_payload = {
        key: value for key, value in implementation.items() if key != "lock_payload_sha256"
    }
    config = implementation["config_payload"]
    records = [
        json.loads(line)
        for line in (PROJECT_ROOT / implementation["corpus"]).read_text().splitlines()
        if line
    ]
    fixture_paths = sorted((evaluation_dir / "raw-fixtures").glob("*.json"))
    fixture_values = [json.loads(path.read_text()) for path in fixture_paths]
    by_id = {fixture["id"]: fixture for fixture in fixture_values}
    result = json.loads(result_path.read_text())
    access = json.loads(access_path.read_text())
    rows = [score(record, by_id[record["id"]]["raw_response"], config) for record in records]
    metrics = aggregate(rows)
    gates = registered_gates(metrics, result, config, access)
    strict_controls = controls(config)
    strict_diagnostic = {
        "raw_choice_semantic_acceptance_rate": mean(
            [row["strict_raw_choice_valid"] for row in rows]
        ),
        "final_choice_semantic_validity_rate": mean(
            [row["strict_final_choice_valid"] for row in rows]
        ),
        "registered_validator_false_positive_count": sum(
            row["raw_semantic_valid"] and not row["strict_raw_choice_valid"] for row in rows
        ),
        **strict_controls,
    }
    fields = (
        "id", "clarification_code", "resolved_action_code", "style_hint",
        "exact_json_parse", "schema_valid", "raw_question", "raw_semantic_valid",
        "raw_structural_valid", "missing_or_repeated_required_anchors",
        "unrequested_anchors", "forbidden_execution_claim_count", "forbidden_field_count",
        "fallback_used", "final_question", "final_semantic_valid",
        "final_question_character_count", "action_code_preserved", "accepted_noncanonical",
    )
    v79 = json.loads((PROJECT_ROOT / implementation["parent_V79_result"]).read_text())
    expected_nodes = [
        {"fixture": name, **node}
        for name, fixture in v79["fixtures"].items()
        for node in fixture["exact"]["policy_nodes"]
        if node["action"].startswith("ask_")
    ]
    checks = {
        "implementation_lock_payload_valid": payload_hash(implementation_payload)
        == implementation["lock_payload_sha256"],
        "exactly_one_fixture_per_locked_record_in_order": bool(
            len(fixture_values) == len(records) == 24
            and [fixture["id"] for fixture in fixture_values] == [record["id"] for record in records]
        ),
        "independent_registered_parser_validator_fallback_reproduced": all(
            all(close(row[field], by_id[row["id"]][field]) for field in fields)
            for row in rows
        ),
        "independent_registered_metrics_reproduced": close(metrics, result["metrics"]),
        "independent_registered_gates_reproduced": gates == result["gates"],
        "frozen_V79_action_nodes_and_values_unchanged": bool(
            result["policy_invariance"]["clarification_nodes"] == expected_nodes
            and all(
                row["original_value"] == row["surface_layer_value"]
                and row["absolute_error"] == 0.0
                for row in result["policy_invariance"]["fixture_values"]
            )
        ),
        "registered_failure_decision_preserves_only_canonical_and_grammar": bool(
            not all(gates.values())
            and not result["passed"]
            and result["decision"]
            == "freeze_V82_failure_and_retain_only_canonical_and_finite_grammar_renderers"
        ),
        "strict_diagnostic_finds_registered_semantic_false_positives": strict_diagnostic[
            "registered_validator_false_positive_count"
        ] > 0,
        "canonical_and_grammar_pass_strict_choice_diagnostic": bool(
            strict_diagnostic["strict_canonical_validity_rate"] == 1.0
            and strict_diagnostic["strict_finite_grammar_validity_rate"] == 1.0
        ),
        "one_load_twenty_four_generations_zero_external_access": bool(
            access["attempt_number"] == 1
            and access["model_load_count"] == 1
            and access["model_generation_count"] == 24
            and all(
                access[key] == 0
                for key in (
                    "API_call_count", "adapter_training_run_count", "human_record_access_count",
                    "original_user_language_access_count", "real_tool_call_count",
                    "external_side_effect_count",
                )
            )
        ),
        "result_attempt_matches_access": result["attempt"] == access,
        "zero_model_or_external_access_during_verification": True,
    }
    passed = all(checks.values())
    audit = {
        "schema_version": "82-local-clarification-surface-outcome-audit",
        "experiment": "v82_local_clarification_surface_independent_outcome_audit",
        "passed": passed,
        "decision": (
            "freeze_verified_failure_and_disallow_local_surface_renderer"
            if passed else "reject_V82_outcome_closure"
        ),
        "checks": checks,
        "registered_metrics": metrics,
        "registered_gates": gates,
        "strict_post_outcome_choice_diagnostic": strict_diagnostic,
        "claim_correction": (
            "registered final_semantic_validity is lexical-anchor validity, not a complete "
            "semantic proof; the stricter choice diagnostic governs reporting"
        ),
        "access": {
            "model_load_count": 0, "model_generation_count": 0, "API_call_count": 0,
            "adapter_training_run_count": 0, "human_record_access_count": 0,
            "original_user_language_access_count": 0, "real_tool_call_count": 0,
            "external_side_effect_count": 0,
        },
    }
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    if not passed:
        print(json.dumps(audit, indent=2, sort_keys=True))
        raise SystemExit(1)
    outcome_lock = {
        "schema_version": "82-local-clarification-surface-outcome-lock",
        "experiment": "v82_local_clarification_surface_outcome_lock",
        "implementation_lock": str(implementation_path.relative_to(PROJECT_ROOT)),
        "implementation_lock_sha256": file_sha256(implementation_path),
        "result": str(result_path.relative_to(PROJECT_ROOT)),
        "result_sha256": file_sha256(result_path),
        "access": str(access_path.relative_to(PROJECT_ROOT)),
        "access_sha256": file_sha256(access_path),
        "raw_fixture_manifest": {
            str(path.relative_to(PROJECT_ROOT)): file_sha256(path) for path in fixture_paths
        },
        "outcome_verifier": str(verifier_path.relative_to(PROJECT_ROOT)),
        "outcome_verifier_sha256": file_sha256(verifier_path),
        "outcome_audit": str(audit_path.relative_to(PROJECT_ROOT)),
        "outcome_audit_sha256": file_sha256(audit_path),
        "outcome": {
            "passed": False,
            "registered_raw_semantic_acceptance_rate": metrics[
                "raw_semantic_acceptance_rate"
            ],
            "registered_fallback_rate": metrics["fallback_rate"],
            "registered_final_anchor_validity_rate": metrics[
                "final_semantic_validity_rate"
            ],
            "strict_raw_choice_semantic_acceptance_rate": strict_diagnostic[
                "raw_choice_semantic_acceptance_rate"
            ],
            "strict_final_choice_semantic_validity_rate": strict_diagnostic[
                "final_choice_semantic_validity_rate"
            ],
            "registered_validator_false_positive_count": strict_diagnostic[
                "registered_validator_false_positive_count"
            ],
            "V79_action_invariance_rate": result["policy_invariance"][
                "reachable_clarification_action_invariance_rate"
            ],
            "V79_maximum_policy_value_absolute_error": result["policy_invariance"][
                "maximum_policy_value_absolute_error"
            ],
        },
        "authorization": {
            "modify_or_rerun_V82": False,
            "run_V82_local_model_again": False,
            "deploy_local_model_surface_renderer": False,
            "use_locked_canonical_renderer": True,
            "use_locked_finite_grammar_renderer": True,
            "continue_local_model_candidate_or_surface_integration": False,
            "run_API_model": False,
            "train_adapter": False,
            "collect_human_or_original_user_language": False,
            "perform_real_tool_call_or_external_side_effect": False,
            "integrate_and_verify_model_free_fail_closed_interface": True,
        },
    }
    outcome_lock["lock_payload_sha256"] = payload_hash(outcome_lock)
    lock_path.write_text(json.dumps(outcome_lock, indent=2, sort_keys=True) + "\n")
    print(json.dumps(audit, indent=2, sort_keys=True))
    print(json.dumps({"lock": str(lock_path), "sha256": file_sha256(lock_path)}, indent=2))


if __name__ == "__main__":
    main()
