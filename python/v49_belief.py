"""Exact passive hidden-state filtering and fresh stochastic registry for V49."""
from __future__ import annotations

import json
import math
from decimal import Decimal, localcontext
from fractions import Fraction
from itertools import product
from typing import Any, Iterable, Sequence

from v22_relational import canonical_json, sha256_text
from v42_stateful import atom_universe, effect, negate, relation, unary, world_signature
from v46_stochastic import (
    PROBABILITIES,
    _apply_payload,
    _condition_holds,
    _configuration_key,
    _deliver,
    _fraction,
    _rule,
    _validate_action,
    canonical_program,
    delayed,
    mechanic_registry as v46_registry,
    program_key,
    stochastic,
)
from v47_sampling import execute_joint_distribution, mechanic_registry as v47_registry
from v48_composition import FAMILIES, mechanic_registry as v48_registry


def _expanded_candidates():
    targets = [
        *(unary(predicate, variable) for predicate, variable in product(("active", "marked", "ready"), ("actor", "target"))),
        relation("actor", "target"),
        relation("target", "actor"),
    ]
    conditions = [
        unary("active", "actor"),
        negate(unary("active", "actor")),
        unary("marked", "target"),
        negate(unary("ready", "target")),
        relation("actor", "target"),
        negate(relation("actor", "target")),
        relation("target", "actor"),
        negate(relation("target", "actor")),
    ]
    for probability, operation, target in product(PROBABILITIES, ("set_true", "set_false", "toggle"), targets):
        yield "immediate_bernoulli_mutation", probability, canonical_program({"rules": [
            _rule("pulse", stochastic_immediate=[stochastic(str(probability), effect(operation, target))]),
            _rule("route", deterministic_immediate=[effect("toggle", relation("actor", "target"))]),
        ]})
    for probability, delay_ticks, operation, target in product(
        PROBABILITIES, (1, 2), ("set_true", "set_false", "toggle"), targets
    ):
        yield "delayed_bernoulli_scheduling", probability, canonical_program({"rules": [
            _rule("pulse", stochastic_delayed=[delayed(
                delay_ticks, stochastic(str(probability), effect(operation, target))
            )]),
            _rule("route", deterministic_immediate=[effect("toggle", unary("marked", "actor"))]),
        ]})
    for probability, condition, target in product(PROBABILITIES, conditions, targets):
        yield "state_conditional_probability", probability, canonical_program({"rules": [
            _rule("pulse", deterministic_immediate=[effect("toggle", unary("active", "actor"))]),
            _rule("route", stochastic_immediate=[stochastic(
                str(probability), effect("toggle", target), condition
            )]),
        ]})
    for probability, delay_ticks, operation, target in product(
        PROBABILITIES, (1, 2), ("set_true", "toggle"), targets
    ):
        yield "interleaved_deterministic_and_stochastic", probability, canonical_program({"rules": [
            _rule("pulse", stochastic_delayed=[delayed(
                delay_ticks, stochastic(str(probability), effect(operation, target))
            )]),
            _rule("route", deterministic_immediate=[effect("toggle", target)]),
        ]})


def mechanic_registry() -> list[dict[str, Any]]:
    excluded = {row["key"] for registry in (v46_registry, v47_registry, v48_registry) for row in registry()}
    grouped = {(family, probability): [] for family in FAMILIES for probability in PROBABILITIES}
    for family, probability, program in _expanded_candidates():
        key = program_key(program)
        if key not in excluded:
            grouped[(family, probability)].append((sha256_text(key), key, program))
    rows: list[dict[str, Any]] = []
    for family in FAMILIES:
        family_ordinal = 0
        for probability in PROBABILITIES:
            candidates = sorted(set((digest, key, canonical_json(program)) for digest, key, program in grouped[(family, probability)]))
            if len(candidates) < 4:
                raise RuntimeError(f"V49 lacks fresh candidates for {family}/{probability}")
            for digest, key, encoded_program in candidates[:4]:
                del digest
                program = json.loads(encoded_program)
                rows.append({
                    "family": family,
                    "ordinal": family_ordinal,
                    "probability": str(probability),
                    "timing": "delayed" if any(rule["stochastic_delayed"] for rule in program["rules"]) else "immediate",
                    "program": program,
                    "key": key,
                    "id": f"mechanic_{sha256_text(key)[:16]}",
                })
                family_ordinal += 1
    keys = {row["key"] for row in rows}
    if len(rows) != 48 or len(keys) != 48 or keys & excluded:
        raise RuntimeError("V49 registry must contain 48 fresh unique programs")
    return rows


def fraction_value(row: dict[str, int]) -> Fraction:
    return Fraction(row["numerator"], row["denominator"])


def trajectory_map(program, entities, initial_world, actions) -> dict[str, Fraction]:
    return {
        canonical_json(row["trajectory"]): fraction_value(row["mass"])
        for row in execute_joint_distribution(program, entities, initial_world, actions)
    }


def signature_world(signature: str) -> dict[str, bool]:
    return {row["atom"]: row["value"] for row in json.loads(signature)}


def masked_trace(trajectory: Sequence[str], masks: Sequence[Sequence[str]]) -> list[list[dict[str, Any]]]:
    if len(trajectory) != len(masks):
        raise ValueError("V49 trajectory and mask lengths differ")
    result = []
    for signature, mask in zip(trajectory, masks, strict=True):
        world = signature_world(signature)
        if not set(mask) <= set(world):
            raise ValueError("V49 mask references an unknown atom")
        result.append([{"atom": atom, "value": world[atom]} for atom in sorted(mask)])
    return result


def trace_key(trace: Sequence[Sequence[dict[str, Any]]]) -> str:
    return canonical_json(trace)


def trajectory_matches(trajectory: Sequence[str], evidence: Sequence[Sequence[dict[str, Any]]]) -> bool:
    if len(evidence) > len(trajectory):
        return False
    for signature, observed in zip(trajectory, evidence, strict=False):
        world = signature_world(signature)
        if any(world.get(row["atom"]) is not row["value"] for row in observed):
            return False
    return True


def observation_distribution(
    program, entities, initial_world, actions, masks
) -> dict[str, Fraction]:
    result: dict[str, Fraction] = {}
    for trajectory_key, mass in trajectory_map(program, entities, initial_world, actions).items():
        key = trace_key(masked_trace(json.loads(trajectory_key), masks))
        result[key] = result.get(key, Fraction(0)) + mass
    if sum(result.values(), Fraction(0)) != 1:
        raise RuntimeError("V49 masked observation distribution does not normalize")
    return result


def conditional_suffix_from_map(
    distribution: dict[str, Fraction], evidence: Sequence[Sequence[dict[str, Any]]], prefix_length: int
) -> tuple[Fraction, dict[str, Fraction]]:
    joint: dict[str, Fraction] = {}
    evidence_mass = Fraction(0)
    for trajectory_key, mass in distribution.items():
        trajectory = json.loads(trajectory_key)
        if trajectory_matches(trajectory[:prefix_length], evidence):
            evidence_mass += mass
            suffix_key = canonical_json(trajectory[prefix_length:])
            joint[suffix_key] = joint.get(suffix_key, Fraction(0)) + mass
    if not evidence_mass:
        return Fraction(0), {}
    return evidence_mass, {key: value / evidence_mass for key, value in joint.items()}


def conditional_suffix_distribution(
    program, entities, initial_world, actions, evidence, prefix_length
) -> tuple[Fraction, dict[str, Fraction]]:
    return conditional_suffix_from_map(
        trajectory_map(program, entities, initial_world, actions), evidence, prefix_length
    )


def _catalog_counts(ids: Iterable[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in ids:
        counts[value] = counts.get(value, 0) + 1
    return counts


def support_posterior(
    registry: Sequence[dict[str, Any]], supports: Sequence[dict[str, Any]], fully_observed: bool = False
) -> list[Decimal]:
    log_weights: list[Decimal | None] = []
    with localcontext() as context:
        context.prec = 100
        for mechanic in registry:
            score = Decimal(0)
            possible = True
            for support in supports:
                initial_world = {row["atom"]: row["allowed_values"][0] for row in support["initial_state"]}
                full = trajectory_map(mechanic["program"], support["entities"], initial_world, support["actions"])
                if fully_observed:
                    catalog = support["full_trajectory_catalog"]
                    probabilities = full
                    ids = support["realized_full_trajectory_ids"]
                else:
                    catalog = support["masked_trace_catalog"]
                    probabilities: dict[str, Fraction] = {}
                    for key, mass in full.items():
                        observation_key = trace_key(masked_trace(json.loads(key), support["masks"]))
                        probabilities[observation_key] = probabilities.get(observation_key, Fraction(0)) + mass
                    ids = support["realized_masked_trace_ids"]
                for outcome_id, count in _catalog_counts(ids).items():
                    probability = probabilities.get(canonical_json(catalog[outcome_id]), Fraction(0))
                    if not probability:
                        possible = False
                        break
                    score += count * (
                        Decimal(probability.numerator).ln() - Decimal(probability.denominator).ln()
                    )
                if not possible:
                    break
            log_weights.append(score if possible else None)
        valid = [value for value in log_weights if value is not None]
        if not valid:
            raise RuntimeError("V49 posterior has zero evidence under every program")
        maximum = max(valid)
        weights = [Decimal(0) if value is None else (value - maximum).exp() for value in log_weights]
        total = sum(weights, Decimal(0))
        return [weight / total for weight in weights]


def query_predictive(
    registry: Sequence[dict[str, Any]],
    support_weights: Sequence[Decimal],
    entities,
    initial_world,
    actions,
    evidence,
    prefix_length: int,
) -> tuple[dict[str, Decimal], list[Decimal], list[dict[str, Fraction]]]:
    with localcontext() as context:
        context.prec = 100
        evidence_masses: list[Fraction] = []
        conditionals: list[dict[str, Fraction]] = []
        unnormalized: list[Decimal] = []
        for mechanic, prior in zip(registry, support_weights, strict=True):
            evidence_mass, conditional = conditional_suffix_distribution(
                mechanic["program"], entities, initial_world, actions, evidence, prefix_length
            )
            evidence_masses.append(evidence_mass)
            conditionals.append(conditional)
            unnormalized.append(prior * Decimal(evidence_mass.numerator) / Decimal(evidence_mass.denominator))
        total = sum(unnormalized, Decimal(0))
        if not total:
            raise RuntimeError("V49 query evidence has zero posterior probability")
        query_weights = [value / total for value in unnormalized]
        mixture: dict[str, Decimal] = {}
        for weight, conditional in zip(query_weights, conditionals, strict=True):
            for key, probability in conditional.items():
                mixture[key] = mixture.get(key, Decimal(0)) + weight * Decimal(probability.numerator) / Decimal(
                    probability.denominator
                )
        if abs(sum(mixture.values(), Decimal(0)) - Decimal(1)) > Decimal("1e-80"):
            raise RuntimeError("V49 query predictive does not normalize")
        return mixture, query_weights, conditionals


def _configuration_key_with_history(world, queue, history) -> str:
    return canonical_json({"configuration": _configuration_key(world, queue), "history": history})


def _accumulate_configuration(target, world, queue, history, mass):
    if not mass:
        return
    key = _configuration_key_with_history(world, queue, history)
    if key not in target:
        target[key] = {"world": dict(world), "queue": list(queue), "history": list(history), "mass": Fraction(0)}
    target[key]["mass"] += mass


def advance_configurations(program, entities, configurations, actions, start_tick: int):
    normalized = canonical_program(program)
    rules = {row["action"]: row for row in normalized["rules"]}
    current = configurations
    for offset, action in enumerate(actions):
        tick = start_tick + offset
        action_id, binding = _validate_action(action, entities)
        successors = {}
        for configuration in current.values():
            world, queue = _deliver(configuration["queue"], tick, configuration["world"], entities)
            mass = configuration["mass"]
            alternatives = [(world, queue, mass)]
            if action_id != "wait":
                rule = rules[action_id]
                base = dict(world)
                for payload in rule["deterministic_immediate"]:
                    base = _apply_payload(payload, base, binding, entities)
                branches = rule["stochastic_immediate"] or rule["stochastic_delayed"]
                if branches and _condition_holds(branches[0], world, binding, entities):
                    branch = branches[0]
                    probability = _fraction(branch["probability"])
                    alternatives = [(base, queue, mass * (1 - probability))]
                    if rule["stochastic_immediate"]:
                        alternatives.append((_apply_payload(branch["effect"], base, binding, entities), queue, mass * probability))
                    else:
                        event = {"due": tick + branch["delay"], "effect": branch["effect"], "binding": dict(binding)}
                        alternatives.append((base, [*queue, event], mass * probability))
                else:
                    alternatives = [(base, queue, mass)]
            for next_world, next_queue, next_mass in alternatives:
                history = [*configuration["history"], world_signature(next_world)]
                _accumulate_configuration(successors, next_world, next_queue, history, next_mass)
        current = successors
    return current


def prefix_configurations(program, entities, initial_world, prefix_actions, evidence):
    initial = {
        _configuration_key_with_history(initial_world, [], []): {
            "world": dict(initial_world), "queue": [], "history": [], "mass": Fraction(1)
        }
    }
    configurations = advance_configurations(program, entities, initial, prefix_actions, 0)
    selected = {key: row for key, row in configurations.items() if trajectory_matches(row["history"], evidence)}
    mass = sum((row["mass"] for row in selected.values()), Fraction(0))
    return mass, selected


def map_latent_predictive(
    registry: Sequence[dict[str, Any]], support_weights: Sequence[Decimal], entities, initial_world,
    actions, evidence, prefix_length: int
) -> dict[str, Decimal]:
    program_rows = []
    unnormalized = []
    with localcontext() as context:
        context.prec = 100
        for mechanic, prior in zip(registry, support_weights, strict=True):
            evidence_mass, configurations = prefix_configurations(
                mechanic["program"], entities, initial_world, actions[:prefix_length], evidence
            )
            unnormalized.append(prior * Decimal(evidence_mass.numerator) / Decimal(evidence_mass.denominator))
            if not configurations:
                program_rows.append({})
                continue
            selected_key = min(configurations, key=lambda key: (-configurations[key]["mass"], key))
            selected = configurations[selected_key]
            seed = {
                _configuration_key_with_history(selected["world"], selected["queue"], []): {
                    "world": selected["world"], "queue": selected["queue"], "history": [], "mass": Fraction(1)
                }
            }
            suffix_configs = advance_configurations(
                mechanic["program"], entities, seed, actions[prefix_length:], prefix_length
            )
            distribution: dict[str, Fraction] = {}
            for row in suffix_configs.values():
                key = canonical_json(row["history"])
                distribution[key] = distribution.get(key, Fraction(0)) + row["mass"]
            program_rows.append(distribution)
        total = sum(unnormalized, Decimal(0))
        if not total:
            raise RuntimeError("V49 MAP-state control has zero evidence")
        weights = [value / total for value in unnormalized]
        result: dict[str, Decimal] = {}
        for weight, distribution in zip(weights, program_rows, strict=True):
            for key, probability in distribution.items():
                result[key] = result.get(key, Decimal(0)) + weight * Decimal(probability.numerator) / Decimal(
                    probability.denominator
                )
        return result


def entropy(distribution: dict[str, Any]) -> float:
    return -sum(float(value) * math.log(float(value)) for value in distribution.values() if value)


def effective_count(weights: Sequence[Decimal]) -> float:
    return math.exp(-sum(float(value) * math.log(float(value)) for value in weights if value))


def posterior_uncertainty(
    predictive: dict[str, Decimal], query_weights: Sequence[Decimal], conditionals: Sequence[dict[str, Fraction]]
) -> dict[str, float]:
    predictive_entropy = entropy(predictive)
    expected_within = sum(float(weight) * entropy(distribution) for weight, distribution in zip(query_weights, conditionals, strict=True))
    return {
        "predictive_entropy": predictive_entropy,
        "expected_within_program_entropy": expected_within,
        "program_suffix_mutual_information": max(0.0, predictive_entropy - expected_within),
        "effective_program_count": effective_count(query_weights),
    }


def fraction_rows(distribution: dict[str, Fraction], field: str = "suffix") -> list[dict[str, Any]]:
    return [
        {field: json.loads(key), "mass": {"numerator": value.numerator, "denominator": value.denominator}}
        for key, value in sorted(distribution.items())
    ]


def decimal_map(distribution: dict[str, Fraction]) -> dict[str, Decimal]:
    return {key: Decimal(value.numerator) / Decimal(value.denominator) for key, value in distribution.items()}


def full_evidence(trajectory: Sequence[str], prefix_length: int) -> list[list[dict[str, Any]]]:
    return [
        [{"atom": atom, "value": value} for atom, value in sorted(signature_world(signature).items())]
        for signature in trajectory[:prefix_length]
    ]


def all_atoms(entities: Sequence[dict[str, str]]) -> tuple[str, ...]:
    return atom_universe(entities)
