#!/usr/bin/env python3
"""Narrow V77r1 repair for terminal policies with omitted terminal branches."""
from __future__ import annotations

from typing import Any

import numpy as np

from v77_clarification_benchmark import (
    ClarificationKernel,
    exact_step,
    is_terminal_belief,
    validate_belief,
)


def complete_terminal_branches(
    kernel: ClarificationKernel,
    belief: np.ndarray,
    policy: dict[str, Any],
    horizon: int,
) -> dict[str, Any]:
    """Add only omitted branches whose successor belief is already terminal.

    The repair deliberately refuses to invent a continuation for an active
    successor. It therefore cannot alter any nonterminal policy decision.
    """
    value = validate_belief(kernel, belief)
    repaired = dict(policy)
    if horizon <= 1 or is_terminal_belief(kernel, value) or policy.get("terminal"):
        return repaired
    action = int(policy["selected_action"])
    step = exact_step(kernel, value, action)
    existing = policy.get("branches", {})
    branches: dict[int, dict[str, Any]] = {}
    for observation, posterior in step["posteriors"].items():
        if observation in existing:
            branches[observation] = complete_terminal_branches(
                kernel,
                posterior,
                existing[observation],
                horizon - 1,
            )
        elif is_terminal_belief(kernel, posterior):
            branches[observation] = {
                "terminal": True,
                "horizon": horizon - 1,
                "value": 0.0,
            }
        else:
            raise RuntimeError(
                "V77r1 repair refuses to synthesize a nonterminal policy branch"
            )
    repaired["branches"] = branches
    return repaired
