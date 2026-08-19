from __future__ import annotations

import math
from collections import Counter
from typing import Any


def phase_class(phase: str) -> str:
    if phase == "ambiguous":
        return "ambiguous"
    if phase.startswith("clarified_"):
        return "clarified"
    return "clear"


def audit_gap(
    result: dict[str, Any],
    hidden_rows: list[dict[str, Any]],
    config: dict[str, Any],
) -> dict[str, Any]:
    hidden = {row["fixture_id"]: row for row in hidden_rows}
    rows = {
        condition: {
            row["fixture_id"]: row
            for row in result["fixtures"].values()
            if row["condition_id"] == condition
        }
        for condition in ("direct", "thinking")
    }
    expected_ids = set(hidden)
    if any(set(condition_rows) != expected_ids for condition_rows in rows.values()):
        raise ValueError("V140 requires exact paired V139 fixture coverage")
    thinking = rows["thinking"]
    direct = rows["direct"]
    failed_gates = sorted(key for key, value in result["summary"]["conditions"]["thinking"]["gates"].items() if not value)
    invalid = [row for row in thinking.values() if not row["response_valid"]]
    ambiguous_ids = [fixture_id for fixture_id, row in hidden.items() if phase_class(row["phase"]) == "ambiguous"]
    valid_ambiguous = [thinking[fixture_id] for fixture_id in ambiguous_ids if thinking[fixture_id]["response_valid"]]
    apparent_ambiguous_correct = sum(thinking[fixture_id]["answer_choice_id"] == hidden[fixture_id]["truth_choice_id"] for fixture_id in ambiguous_ids)
    valid_ambiguous_correct = sum(row["answer_choice_id"] == hidden[row["fixture_id"]]["truth_choice_id"] for row in valid_ambiguous)
    paired = Counter()
    paired_by_phase = Counter()
    for fixture_id, truth in hidden.items():
        direct_correct = direct[fixture_id]["answer_choice_id"] == truth["truth_choice_id"]
        thinking_correct = thinking[fixture_id]["answer_choice_id"] == truth["truth_choice_id"]
        paired[(direct_correct, thinking_correct)] += 1
        paired_by_phase[(phase_class(truth["phase"]), direct_correct, thinking_correct)] += 1

    thresholds = config["registeredThresholds"]
    population = config["registeredPopulation"]
    required_valid = math.ceil(thresholds["minimumStructuredValidity"] * population["fixtureCount"] - 1e-12)
    required_ambiguous = math.ceil(thresholds["minimumAmbiguousAbstentionAccuracy"] * population["ambiguousFixtureCount"] - 1e-12)
    observed_valid = population["fixtureCount"] - len(invalid)
    completion_only_valid = population["fixtureCount"]
    completion_only_ambiguous = apparent_ambiguous_correct
    valid_semantic_errors = [
        row for row in valid_ambiguous
        if row["answer_choice_id"] != hidden[row["fixture_id"]]["truth_choice_id"]
    ]
    semantic_only_ambiguous = apparent_ambiguous_correct + len(valid_semantic_errors)
    semantic_only_valid = observed_valid
    return {
        "failed_thinking_gate_families": failed_gates,
        "invalid_output_count": len(invalid),
        "invalid_reason_counts": dict(sorted(Counter(row["validation_reason"] for row in invalid).items())),
        "invalid_phase_counts": dict(sorted(Counter(phase_class(hidden[row["fixture_id"]]["phase"]) for row in invalid).items())),
        "all_invalid_at_condition_token_ceiling": bool(
            invalid and all(row["generated_token_count"] == 1024 for row in invalid)
        ),
        "structured_validity": {
            "required_valid_count": required_valid,
            "observed_valid_count": observed_valid,
            "minimum_additional_valid_outputs": max(0, required_valid - observed_valid),
        },
        "ambiguity": {
            "required_correct_count": required_ambiguous,
            "apparent_correct_count_with_safe_invalid_fallback": apparent_ambiguous_correct,
            "minimum_additional_apparent_correct": max(0, required_ambiguous - apparent_ambiguous_correct),
            "valid_fixture_count": len(valid_ambiguous),
            "valid_correct_count": valid_ambiguous_correct,
            "valid_only_accuracy": valid_ambiguous_correct / len(valid_ambiguous),
            "valid_semantic_overcommitment_count": len(valid_semantic_errors),
        },
        "paired_correctness": {
            "both_correct": paired[(True, True)],
            "thinking_repairs_direct": paired[(False, True)],
            "thinking_regresses_direct": paired[(True, False)],
            "both_wrong": paired[(False, False)],
            "net_thinking_repairs": paired[(False, True)] - paired[(True, False)],
        },
        "paired_by_phase": {
            f"{phase}|direct={direct_correct}|thinking={thinking_correct}": count
            for (phase, direct_correct, thinking_correct), count in sorted(paired_by_phase.items())
        },
        "counterfactuals": {
            "completion_only": {
                "valid_count": completion_only_valid,
                "ambiguous_correct_count": completion_only_ambiguous,
                "qualifies_both_failed_families": completion_only_valid >= required_valid and completion_only_ambiguous >= required_ambiguous,
            },
            "semantic_only": {
                "valid_count": semantic_only_valid,
                "ambiguous_correct_count": semantic_only_ambiguous,
                "qualifies_both_failed_families": semantic_only_valid >= required_valid and semantic_only_ambiguous >= required_ambiguous,
            },
        },
        "minimum_joint_gap": {
            "fewer_invalid_outputs_required": max(0, required_valid - observed_valid),
            "additional_apparent_ambiguous_corrections_required": max(0, required_ambiguous - apparent_ambiguous_correct),
            "both_mechanism_families_required": True,
        },
    }


__all__ = ["audit_gap", "phase_class"]
