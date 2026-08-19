from __future__ import annotations

from copy import deepcopy
from typing import Any


def child_action(policy: dict[str, Any]) -> str:
    if policy["action"] == "STOP":
        return "STOP"
    return policy["action"]


def corrected_case_is_history_dependent(case: dict[str, Any]) -> bool:
    root = case["exact_bayes_policy"]
    if root["action"] == "STOP":
        return False
    return len({child_action(value["policy"]) for value in root["children"].values()}) > 1


def corrected_history_count(cases: list[dict[str, Any]]) -> int:
    return sum(corrected_case_is_history_dependent(case) for case in cases)


def corrected_summary(evaluation: dict[str, Any]) -> dict[str, Any]:
    summary = deepcopy(evaluation["summary"])
    summary["history_dependent_second_action_case_count"] = corrected_history_count(
        evaluation["cases"]
    )
    return summary


__all__ = [
    "child_action",
    "corrected_case_is_history_dependent",
    "corrected_history_count",
    "corrected_summary",
]
