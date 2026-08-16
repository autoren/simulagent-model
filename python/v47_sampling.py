"""Joint trajectory semantics and fixed Bayesian estimator for V47."""
from __future__ import annotations

import math
import random
from decimal import Decimal, localcontext
from fractions import Fraction
from itertools import product
from typing import Any, Sequence

from v22_relational import canonical_json, sha256_text
from v42_stateful import effect, negate, relation, unary, world_signature
from v42_stateful import compatible_worlds
from v46_stochastic import (
    PROBABILITIES, _accumulate, _apply_payload, _condition_holds, _configuration_key,
    _deliver, _fraction, _rule, _validate_action, canonical_program, delayed,
    mechanic_registry as v46_registry, program_key, stochastic,
)


def _joint_key(world, queue, history):
    return canonical_json({"configuration": _configuration_key(world, queue), "history": history})


def _joint_accumulate(target, world, queue, history, mass):
    if not mass:
        return
    key = _joint_key(world, queue, history)
    if key not in target:
        target[key] = {"world": dict(world), "queue": list(queue), "history": list(history), "mass": Fraction(0)}
    target[key]["mass"] += mass


def execute_joint_distribution(program, entities, initial_world, actions):
    normalized = canonical_program(program)
    rules = {row["action"]: row for row in normalized["rules"]}
    configurations = {_joint_key(initial_world, [], []): {"world": dict(initial_world), "queue": [], "history": [], "mass": Fraction(1)}}
    for tick, action in enumerate(actions):
        action_id, binding = _validate_action(action, entities)
        successors = {}
        for configuration in configurations.values():
            world, queue = _deliver(configuration["queue"], tick, configuration["world"], entities)
            mass = configuration["mass"]
            alternatives = [(world, queue, mass)]
            if action_id != "wait":
                rule = rules[action_id]
                base = dict(world)
                for payload in rule["deterministic_immediate"]:
                    base = _apply_payload(payload, base, binding, entities)
                branch_rows = rule["stochastic_immediate"] or rule["stochastic_delayed"]
                if branch_rows and _condition_holds(branch_rows[0], world, binding, entities):
                    branch = branch_rows[0]
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
                _joint_accumulate(successors, next_world, next_queue, history, next_mass)
        configurations = successors
    rows = [{
        "trajectory": row["history"],
        "mass": {"numerator": row["mass"].numerator, "denominator": row["mass"].denominator},
    } for row in configurations.values()]
    combined = {}
    for row in rows:
        key = canonical_json(row["trajectory"])
        mass = Fraction(row["mass"]["numerator"], row["mass"]["denominator"])
        combined[key] = combined.get(key, Fraction(0)) + mass
    result = [{"trajectory": json_trajectory(key), "mass": fraction_row(mass)} for key, mass in sorted(combined.items())]
    if sum((Fraction(row["mass"]["numerator"], row["mass"]["denominator"]) for row in result), Fraction(0)) != 1:
        raise RuntimeError("V47 joint trajectory mass does not normalize")
    return result


def json_trajectory(key):
    import json
    return json.loads(key)


def fraction_row(value):
    return {"numerator": value.numerator, "denominator": value.denominator}


def joint_map(distribution):
    return {canonical_json(row["trajectory"]): Fraction(row["mass"]["numerator"], row["mass"]["denominator"]) for row in distribution}


def sample_trajectory(program, entities, initial_world, actions, seed):
    rng = random.Random(seed)
    normalized = canonical_program(program)
    rules = {row["action"]: row for row in normalized["rules"]}
    world, queue, history = dict(initial_world), [], []
    for tick, action in enumerate(actions):
        world, queue = _deliver(queue, tick, world, entities)
        action_id, binding = _validate_action(action, entities)
        if action_id != "wait":
            rule = rules[action_id]
            before = dict(world)
            for payload in rule["deterministic_immediate"]:
                world = _apply_payload(payload, world, binding, entities)
            branches = rule["stochastic_immediate"] or rule["stochastic_delayed"]
            if branches and _condition_holds(branches[0], before, binding, entities):
                branch = branches[0]
                if rng.random() < float(_fraction(branch["probability"])):
                    if rule["stochastic_immediate"]:
                        world = _apply_payload(branch["effect"], world, binding, entities)
                    else:
                        queue.append({"due": tick + branch["delay"], "effect": branch["effect"], "binding": dict(binding)})
        history.append(world_signature(world))
    return history


def trial_seed(sampling_seed, mechanic_id, case_id, trial_index):
    return int(sha256_text(f"v47|{sampling_seed}|{mechanic_id}|{case_id}|{trial_index}"), 16)


def _candidate_rows():
    rows = []
    targets = [unary(predicate, variable) for predicate, variable in product(("active", "marked", "ready"), ("actor", "target"))]
    conditions = [unary("active", "actor"), negate(unary("active", "actor")), unary("marked", "target"), relation("actor", "target"), negate(relation("actor", "target"))]
    for probability, operation, target in product(PROBABILITIES, ("set_true", "set_false", "toggle"), targets):
        rows.append(("immediate_bernoulli_mutation", probability, canonical_program({"rules": [_rule("pulse", stochastic_immediate=[stochastic(str(probability), effect(operation, target))]), _rule("route", deterministic_immediate=[effect("toggle", relation("actor", "target"))])]})))
    for probability, delay_ticks, operation, target in product(PROBABILITIES, (1, 2), ("set_true", "set_false", "toggle"), (relation("actor", "target"), unary("ready", "target"))):
        rows.append(("delayed_bernoulli_scheduling", probability, canonical_program({"rules": [_rule("pulse", stochastic_delayed=[delayed(delay_ticks, stochastic(str(probability), effect(operation, target)))]), _rule("route", deterministic_immediate=[effect("toggle", unary("marked", "actor"))])]})))
    # Toggle guarantees that a satisfied condition produces an observable branch;
    # set/clear candidates can be semantic no-ops when they repeat the condition.
    for probability, condition, operation, target in product(PROBABILITIES, conditions, ("toggle",), targets):
        rows.append(("state_conditional_probability", probability, canonical_program({"rules": [_rule("pulse", deterministic_immediate=[effect("toggle", unary("active", "actor"))]), _rule("route", stochastic_immediate=[stochastic(str(probability), effect(operation, target), condition)])]})))
    for probability, delay_ticks, operation, predicate, variable in product(PROBABILITIES, (1, 2), ("set_true", "toggle"), ("active", "marked", "ready"), ("actor", "target")):
        target = unary(predicate, variable)
        rows.append(("interleaved_deterministic_and_stochastic", probability, canonical_program({"rules": [_rule("pulse", stochastic_delayed=[delayed(delay_ticks, stochastic(str(probability), effect(operation, target)))]), _rule("route", deterministic_immediate=[effect("toggle", target)])]})))
    return rows


def mechanic_registry():
    excluded = {row["key"] for row in v46_registry()}
    grouped = {(family, probability): [] for family in ("immediate_bernoulli_mutation", "delayed_bernoulli_scheduling", "state_conditional_probability", "interleaved_deterministic_and_stochastic") for probability in PROBABILITIES}
    for family, probability, program in _candidate_rows():
        key = program_key(program)
        if key not in excluded:
            grouped[(family, probability)].append((sha256_text(key), key, program))
    mechanics = []
    for family in ("immediate_bernoulli_mutation", "delayed_bernoulli_scheduling", "state_conditional_probability", "interleaved_deterministic_and_stochastic"):
        family_ordinal = 0
        for probability in PROBABILITIES:
            for _, key, program in sorted(grouped[(family, probability)])[:4]:
                mechanics.append({"family": family, "ordinal": family_ordinal, "probability": str(probability), "timing": "delayed" if any(rule["stochastic_delayed"] for rule in program["rules"]) else "immediate", "program": program, "key": key, "id": f"mechanic_{sha256_text(key)[:16]}"})
                family_ordinal += 1
    if len(mechanics) != 48 or len({row["key"] for row in mechanics}) != 48 or excluded & {row["key"] for row in mechanics}:
        raise RuntimeError("V47 requires 48 unique programs disjoint from V46")
    return mechanics


def posterior(registry, supports, budget):
    log_weights = []
    with localcontext() as context:
        context.prec = 100
        for mechanic in registry:
            score = Decimal(0)
            possible = True
            for support in supports:
                worlds = compatible_worlds(support["initial_state"])
                if len(worlds) != 1:
                    raise ValueError("V47 support requires a complete initial state")
                probabilities = joint_map(execute_joint_distribution(mechanic["program"], support["entities"], worlds[0], support["actions"]))
                catalog = support["outcome_catalog"]
                counts: dict[str, int] = {}
                for outcome_id in support["realized_outcome_ids"][:budget]:
                    counts[outcome_id] = counts.get(outcome_id, 0) + 1
                for outcome_id, count in counts.items():
                    probability = probabilities.get(canonical_json(catalog[outcome_id]), Fraction(0))
                    if not probability:
                        possible = False
                        break
                    log_probability = Decimal(probability.numerator).ln() - Decimal(probability.denominator).ln()
                    score += count * log_probability
                if not possible:
                    break
            log_weights.append(score if possible else None)
        valid = [value for value in log_weights if value is not None]
        if not valid:
            raise RuntimeError("V47 posterior has zero evidence under every program")
        maximum = max(valid)
        weights = [Decimal(0) if value is None else (value - maximum).exp() for value in log_weights]
        total = sum(weights, Decimal(0))
        return [value / total for value in weights]


def posterior_predictive(registry, weights, entities, initial_world, actions):
    mixture = {}
    with localcontext() as context:
        context.prec = 100
        for mechanic, weight in zip(registry, weights, strict=True):
            if not weight:
                continue
            for key, probability in joint_map(execute_joint_distribution(mechanic["program"], entities, initial_world, actions)).items():
                decimal_probability = Decimal(probability.numerator) / Decimal(probability.denominator)
                mixture[key] = mixture.get(key, Decimal(0)) + weight * decimal_probability
    return mixture
