"""Exact history-dependent belief utilities and fresh stochastic registry for V50."""
from __future__ import annotations

import json
from decimal import Decimal
from fractions import Fraction
from typing import Any, Sequence

from v22_relational import canonical_json, sha256_text
from v46_stochastic import PROBABILITIES, mechanic_registry as v46_registry, program_key
from v47_sampling import mechanic_registry as v47_registry
from v48_composition import FAMILIES, mechanic_registry as v48_registry
from v49_belief import (
    _expanded_candidates,
    conditional_suffix_distribution,
    decimal_map,
    effective_count,
    fraction_rows,
    full_evidence,
    map_latent_predictive,
    masked_trace,
    posterior_uncertainty,
    prefix_configurations,
    query_predictive,
    support_posterior,
    trace_key,
    trajectory_map,
)
from v49_belief import mechanic_registry as v49_registry


def mechanic_registry() -> list[dict[str, Any]]:
    """Return 48 fresh programs, balanced by family and declared probability."""
    excluded = {
        row["key"]
        for registry in (v46_registry, v47_registry, v48_registry, v49_registry)
        for row in registry()
    }
    grouped = {(family, probability): [] for family in FAMILIES for probability in PROBABILITIES}
    for family, probability, program in _expanded_candidates():
        key = program_key(program)
        if key not in excluded:
            grouped[(family, probability)].append((sha256_text(key), key, canonical_json(program)))

    rows: list[dict[str, Any]] = []
    for family in FAMILIES:
        family_ordinal = 0
        for probability in PROBABILITIES:
            candidates = sorted(set(grouped[(family, probability)]))
            if len(candidates) < 4:
                raise RuntimeError(f"V50 lacks fresh candidates for {family}/{probability}")
            for _, key, encoded_program in candidates[:4]:
                program = json.loads(encoded_program)
                stochastic_rule = next(
                    rule for rule in program["rules"]
                    if rule["stochastic_immediate"] or rule["stochastic_delayed"]
                )
                delayed_rows = stochastic_rule["stochastic_delayed"]
                rows.append({
                    "family": family,
                    "ordinal": family_ordinal,
                    "probability": str(probability),
                    "timing": "delayed" if delayed_rows else "immediate",
                    "delay": delayed_rows[0]["delay"] if delayed_rows else 0,
                    "program": program,
                    "key": key,
                    "id": f"mechanic_{sha256_text(key)[:16]}",
                })
                family_ordinal += 1
    keys = {row["key"] for row in rows}
    if len(rows) != 48 or len(keys) != 48 or keys & excluded:
        raise RuntimeError("V50 registry must contain 48 fresh unique programs")
    return rows


def latest_only_evidence(evidence: Sequence[Sequence[dict[str, Any]]]):
    """Retain the observation at the latest prefix time and discard earlier evidence."""
    if not evidence:
        raise ValueError("V50 evidence must contain at least one prefix step")
    return [[] for _ in evidence[:-1]] + [list(evidence[-1])]


def time_shuffled_evidence(
    evidence: Sequence[Sequence[dict[str, Any]]], informative_step: int
) -> list[list[dict[str, Any]]]:
    """Assign informative facts to an adjacent wrong time while preserving values."""
    if not 0 <= informative_step < len(evidence) - 1:
        raise ValueError("V50 informative step must precede the latest prefix step")
    result = [[] for _ in evidence]
    result[-1] = [dict(row) for row in evidence[-1]]
    destination = informative_step - 1 if informative_step else informative_step + 1
    by_atom = {row["atom"]: dict(row) for row in result[destination]}
    for row in evidence[informative_step]:
        by_atom[row["atom"]] = dict(row)
    result[destination] = [by_atom[atom] for atom in sorted(by_atom)]
    return result


def safe_query_predictive(
    registry,
    support_weights,
    entities,
    initial_world,
    actions,
    evidence,
    prefix_length: int,
) -> tuple[dict[str, Decimal], list[Decimal], list[dict[str, Fraction]]]:
    """Return an empty predictive when a deliberately mistimed history is impossible."""
    try:
        return query_predictive(
            registry, support_weights, entities, initial_world, actions, evidence, prefix_length
        )
    except RuntimeError as error:
        if "zero posterior probability" not in str(error):
            raise
        return {}, [Decimal(0) for _ in registry], [{} for _ in registry]


def total_variation(left: dict[str, Any], right: dict[str, Any]) -> float:
    keys = set(left) | set(right)
    return float(sum(abs(Decimal(left.get(key, 0)) - Decimal(right.get(key, 0))) for key in keys) / 2)


def kl_divergence(reference: dict[str, Any], challenger: dict[str, Any]) -> float:
    """Compute KL(reference || challenger), retaining infinity for missing support."""
    total = 0.0
    for key, probability in reference.items():
        p = float(probability)
        if not p:
            continue
        q = float(challenger.get(key, 0))
        if q <= 0:
            return float("inf")
        import math

        total += p * math.log(p / q)
    return total
