"""Fresh registry, independent exact filtering, and SBC utilities for V51."""
from __future__ import annotations

import json
import math
import random
from decimal import Decimal, localcontext
from fractions import Fraction
from typing import Any, Sequence

from v22_relational import canonical_json, sha256_text
from v42_stateful import atom_universe, world_signature
from v46_stochastic import PROBABILITIES, _configuration_key, mechanic_registry as v46_registry, program_key
from v47_sampling import mechanic_registry as v47_registry
from v48_composition import FAMILIES, mechanic_registry as v48_registry
from v49_belief import (
    _accumulate_configuration,
    _configuration_key_with_history,
    _expanded_candidates,
    advance_configurations,
    mechanic_registry as v49_registry,
    prefix_configurations,
    query_predictive,
    support_posterior,
)
from v50_belief import mechanic_registry as v50_registry


def mechanic_registry() -> list[dict[str, Any]]:
    excluded = {
        row["key"]
        for source in (v46_registry, v47_registry, v48_registry, v49_registry, v50_registry)
        for row in source()
    }
    grouped = {(family, probability): [] for family in FAMILIES for probability in PROBABILITIES}
    for family, probability, program in _expanded_candidates():
        key = program_key(program)
        if key not in excluded:
            grouped[(family, probability)].append((sha256_text(key), key, canonical_json(program)))
    rows = []
    for family in FAMILIES:
        family_ordinal = 0
        for probability in PROBABILITIES:
            candidates = sorted(set(grouped[(family, probability)]))
            if len(candidates) < 4:
                raise RuntimeError(f"V51 lacks fresh candidates for {family}/{probability}")
            for _, key, encoded in candidates[:4]:
                program = json.loads(encoded)
                stochastic_rule = next(
                    rule for rule in program["rules"]
                    if rule["stochastic_immediate"] or rule["stochastic_delayed"]
                )
                delayed = stochastic_rule["stochastic_delayed"]
                rows.append({
                    "family": family,
                    "ordinal": family_ordinal,
                    "probability": str(probability),
                    "probability_ordinal": ["1/4", "1/2", "3/4"].index(str(probability)),
                    "timing": "delayed" if delayed else "immediate",
                    "delay": delayed[0]["delay"] if delayed else 0,
                    "program": program,
                    "key": key,
                    "id": f"mechanic_{sha256_text(key)[:16]}",
                })
                family_ordinal += 1
    rows.sort(key=lambda row: row["key"])
    for index, row in enumerate(rows):
        row["program_ordinal"] = index
    keys = {row["key"] for row in rows}
    if len(rows) != 48 or len(keys) != 48 or keys & excluded:
        raise RuntimeError("V51 registry must contain 48 fresh unique programs")
    return rows


def observation_matches(world, observed):
    return all(world.get(row["atom"]) is row["value"] for row in observed)


def sequential_filter(program, entities, initial_world, actions, evidence):
    """Stepwise forward filter that never constructs a full trajectory catalog."""
    initial = {
        _configuration_key_with_history(initial_world, [], []): {
            "world": dict(initial_world), "queue": [], "history": [], "mass": Fraction(1)
        }
    }
    current = initial
    for tick, (action, observed) in enumerate(zip(actions, evidence, strict=True)):
        current = advance_configurations(program, entities, current, [action], tick)
        current = {
            key: row for key, row in current.items()
            if observation_matches(row["world"], observed)
        }
        if not current:
            return Fraction(0), {}
    return sum((row["mass"] for row in current.values()), Fraction(0)), current


def _normalize_decimal(values):
    total = sum(values, Decimal(0))
    if not total:
        raise RuntimeError("V51 exact posterior has zero total mass")
    return [value / total for value in values]


def independent_support_posterior(registry, supports, likelihood_power=1):
    log_weights = []
    with localcontext() as context:
        context.prec = 100
        for mechanic in registry:
            score = Decimal(0)
            possible = True
            for support in supports:
                world = {row["atom"]: row["allowed_values"][0] for row in support["initial_state"]}
                likelihood, _ = sequential_filter(
                    mechanic["program"], support["entities"], world,
                    support["actions"], support["observations"],
                )
                if not likelihood:
                    possible = False
                    break
                score += likelihood_power * (
                    Decimal(likelihood.numerator).ln() - Decimal(likelihood.denominator).ln()
                )
            log_weights.append(score if possible else None)
        valid = [value for value in log_weights if value is not None]
        if not valid:
            raise RuntimeError("V51 every program has zero support likelihood")
        maximum = max(valid)
        raw = [Decimal(0) if value is None else (value - maximum).exp() for value in log_weights]
        return _normalize_decimal(raw)


def batch_support_posterior(registry, supports):
    encoded = []
    for support in supports:
        encoded.append({
            **support,
            "masked_trace_catalog": {"observed": support["observations"]},
            "realized_masked_trace_ids": ["observed"],
        })
    return support_posterior(registry, encoded, fully_observed=False)


def _joint_key(program_key_value, configuration_key):
    return canonical_json({"program": program_key_value, "configuration": configuration_key})


def _configuration_marginal(joint, metadata):
    result = {}
    for key, mass in joint.items():
        configuration = metadata[key]["configuration_key"]
        result[configuration] = result.get(configuration, Decimal(0)) + mass
    return result


def joint_belief_from_filters(registry, support_weights, query, stepwise=False):
    prefix = query["prefix_length"]
    world = {row["atom"]: row["allowed_values"][0] for row in query["initial_state"]}
    raw, metadata, program_raw = {}, {}, [Decimal(0) for _ in registry]
    filters = []
    with localcontext() as context:
        context.prec = 100
        for index, (mechanic, prior) in enumerate(zip(registry, support_weights, strict=True)):
            if stepwise:
                _, configurations = sequential_filter(
                    mechanic["program"], query["entities"], world,
                    query["actions"][:prefix], query["observations"],
                )
            else:
                _, configurations = prefix_configurations(
                    mechanic["program"], query["entities"], world,
                    query["actions"][:prefix], query["observations"],
                )
            filters.append(configurations)
            for row in configurations.values():
                config_key = _configuration_key(row["world"], row["queue"])
                key = _joint_key(mechanic["key"], config_key)
                value = prior * Decimal(row["mass"].numerator) / Decimal(row["mass"].denominator)
                raw[key] = raw.get(key, Decimal(0)) + value
                program_raw[index] += value
                metadata[key] = {
                    "program_index": index,
                    "program_key": mechanic["key"],
                    "program_ordinal": mechanic["program_ordinal"],
                    "probability_ordinal": mechanic["probability_ordinal"],
                    "configuration_key": config_key,
                }
        total = sum(raw.values(), Decimal(0))
        if not total:
            raise RuntimeError("V51 query evidence has zero joint mass")
        joint = {key: value / total for key, value in raw.items()}
        program = [value / total for value in program_raw]
        return joint, metadata, program, filters


def independent_suffix_predictive(registry, support_weights, query, joint, metadata, filters):
    prefix = query["prefix_length"]
    result = {}
    with localcontext() as context:
        context.prec = 100
        for mechanic, configurations in zip(registry, filters, strict=True):
            for row in configurations.values():
                config_key = _configuration_key(row["world"], row["queue"])
                key = _joint_key(mechanic["key"], config_key)
                joint_mass = joint.get(key, Decimal(0))
                if not joint_mass:
                    continue
                same_config_mass = sum(
                    candidate["mass"] for candidate in configurations.values()
                    if _configuration_key(candidate["world"], candidate["queue"]) == config_key
                )
                conditional_history_mass = row["mass"] / same_config_mass
                seed = {
                    _configuration_key_with_history(row["world"], row["queue"], []): {
                        "world": dict(row["world"]), "queue": list(row["queue"]),
                        "history": [], "mass": Fraction(1),
                    }
                }
                suffix = advance_configurations(
                    mechanic["program"], query["entities"], seed,
                    query["actions"][prefix:], prefix,
                )
                for suffix_row in suffix.values():
                    suffix_key = canonical_json(suffix_row["history"])
                    probability = (
                        joint_mass
                        * Decimal(conditional_history_mass.numerator) / Decimal(conditional_history_mass.denominator)
                        * Decimal(suffix_row["mass"].numerator) / Decimal(suffix_row["mass"].denominator)
                    )
                    result[suffix_key] = result.get(suffix_key, Decimal(0)) + probability
        total = sum(result.values(), Decimal(0))
        return {key: value / total for key, value in result.items()}


def batch_inference(registry, supports, query):
    support_weights = batch_support_posterior(registry, supports)
    joint, metadata, query_program, filters = joint_belief_from_filters(
        registry, support_weights, query, stepwise=False
    )
    world = {row["atom"]: row["allowed_values"][0] for row in query["initial_state"]}
    suffix, batch_query_program, _ = query_predictive(
        registry, support_weights, query["entities"], world, query["actions"],
        query["observations"], query["prefix_length"],
    )
    return {
        "support_program": support_weights,
        "query_program": batch_query_program,
        "joint": joint,
        "metadata": metadata,
        "configuration": _configuration_marginal(joint, metadata),
        "suffix": suffix,
    }


def independent_inference(registry, supports, query, likelihood_power=1):
    support_weights = independent_support_posterior(registry, supports, likelihood_power)
    joint, metadata, query_program, filters = joint_belief_from_filters(
        registry, support_weights, query, stepwise=True
    )
    suffix = independent_suffix_predictive(
        registry, support_weights, query, joint, metadata, filters
    )
    return {
        "support_program": support_weights,
        "query_program": query_program,
        "joint": joint,
        "metadata": metadata,
        "configuration": _configuration_marginal(joint, metadata),
        "suffix": suffix,
    }


def categorical_sample(distribution: dict[str, Any], seed: int):
    rng = random.Random(seed)
    draw = rng.random()
    cumulative = 0.0
    for key, probability in sorted(distribution.items()):
        cumulative += float(probability)
        if draw < cumulative:
            return key
    return sorted(distribution)[-1]


def randomized_rank(true_value, draw_values, seed: int):
    less = sum(value < true_value for value in draw_values)
    equal = sum(value == true_value for value in draw_values)
    return less + random.Random(seed).randint(0, equal)


def distribution_tv(left, right):
    keys = set(left) | set(right)
    return float(sum(abs(Decimal(left.get(key, 0)) - Decimal(right.get(key, 0))) for key in keys) / 2)


def sequence_tv(left: Sequence[Decimal], right: Sequence[Decimal]):
    return float(sum(abs(a - b) for a, b in zip(left, right, strict=True)) / 2)
