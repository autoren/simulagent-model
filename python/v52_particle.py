"""Rao-Blackwellized bounded particle filtering utilities for V52."""
from __future__ import annotations

import json
import random
from decimal import Decimal, localcontext
from fractions import Fraction
from itertools import product
from typing import Any, Sequence

from v22_relational import canonical_json, sha256_text
from v42_stateful import effect, relation, unary
from v46_stochastic import (
    PROBABILITIES,
    _configuration_key,
    _rule,
    canonical_program,
    delayed,
    mechanic_registry as v46_registry,
    program_key,
    stochastic,
)
from v47_sampling import mechanic_registry as v47_registry
from v48_composition import mechanic_registry as v48_registry
from v49_belief import (
    _configuration_key_with_history,
    _expanded_candidates,
    advance_configurations,
    mechanic_registry as v49_registry,
)
from v50_belief import mechanic_registry as v50_registry
from v51_sbc import mechanic_registry as v51_registry


V52_FAMILIES = (
    "state_conditional_probability",
    "immediate_vs_delayed_stochasticity",
    "cross_action_stochastic_composition",
    "relational_stochastic_effects",
)


def _v52_candidates():
    for family, probability, program in _expanded_candidates():
        if family == "state_conditional_probability":
            yield "state_conditional_probability", probability, program
        elif family == "interleaved_deterministic_and_stochastic":
            yield "cross_action_stochastic_composition", probability, program

    unary_targets = [
        unary(predicate, variable)
        for predicate, variable in product(("active", "marked", "ready"), ("actor", "target"))
    ]
    relation_targets = [relation("actor", "target"), relation("target", "actor")]
    for probability, operation, target, deterministic_target in product(
        PROBABILITIES,
        ("set_true", "set_false", "toggle"),
        unary_targets,
        relation_targets,
    ):
        yield "immediate_vs_delayed_stochasticity", probability, canonical_program({"rules": [
            _rule("pulse", stochastic_immediate=[
                stochastic(str(probability), effect(operation, target))
            ]),
            _rule("route", deterministic_immediate=[effect("toggle", deterministic_target)]),
        ]})
        yield "immediate_vs_delayed_stochasticity", probability, canonical_program({"rules": [
            _rule("pulse", stochastic_delayed=[
                delayed(2, stochastic(str(probability), effect(operation, target)))
            ]),
            _rule("route", deterministic_immediate=[effect("toggle", deterministic_target)]),
        ]})

    deterministic_targets = [
        unary(predicate, variable)
        for predicate, variable in product(("active", "marked", "ready"), ("actor", "target"))
    ]
    for probability, operation, target, deterministic_target, timing in product(
        PROBABILITIES,
        ("set_true", "set_false", "toggle"),
        relation_targets,
        deterministic_targets,
        ("immediate", "delayed"),
    ):
        branch = stochastic(str(probability), effect(operation, target))
        stochastic_rule = (
            _rule("pulse", stochastic_immediate=[branch])
            if timing == "immediate"
            else _rule("pulse", stochastic_delayed=[delayed(1, branch)])
        )
        yield "relational_stochastic_effects", probability, canonical_program({"rules": [
            stochastic_rule,
            _rule("route", deterministic_immediate=[effect("toggle", deterministic_target)]),
        ]})


def mechanic_registry() -> list[dict[str, Any]]:
    excluded = {
        row["key"]
        for source in (
            v46_registry, v47_registry, v48_registry, v49_registry, v50_registry, v51_registry
        )
        for row in source()
    }
    grouped = {
        (family, probability): [] for family in V52_FAMILIES for probability in PROBABILITIES
    }
    for family, probability, program in _v52_candidates():
        key = program_key(program)
        if key not in excluded:
            grouped[(family, probability)].append((sha256_text(key), key, canonical_json(program)))
    rows = []
    for family in V52_FAMILIES:
        family_ordinal = 0
        for probability in PROBABILITIES:
            candidates = sorted(set(grouped[(family, probability)]))
            if len(candidates) < 4:
                raise RuntimeError(f"V52 lacks fresh candidates for {family}/{probability}")
            for _, key, encoded in candidates[:4]:
                program = json.loads(encoded)
                stochastic = next(
                    rule for rule in program["rules"]
                    if rule["stochastic_immediate"] or rule["stochastic_delayed"]
                )
                delayed = stochastic["stochastic_delayed"]
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
        raise RuntimeError("V52 registry must contain 48 fresh unique programs")
    return rows


def observation_matches(world, observed):
    return all(world.get(row["atom"]) is row["value"] for row in observed)


def stream_seed(base_seed: int, *parts: Any) -> int:
    return int(sha256_text("|".join(["v52-stream", str(base_seed), *(str(part) for part in parts)])), 16)


def stream_id(base_seed: int, *parts: Any) -> str:
    return sha256_text("|".join(["v52-stream", str(base_seed), *(str(part) for part in parts)]))


def _unit_transition(program, entities, world, queue, action, tick):
    seed = {
        _configuration_key_with_history(world, queue, []): {
            "world": dict(world), "queue": list(queue), "history": [], "mass": Fraction(1)
        }
    }
    return advance_configurations(program, entities, seed, [action], tick)


def _merge_groups(groups):
    merged = {}
    for row in groups:
        key = (row["configuration_key"], str(row["individual_weight"]))
        if key not in merged:
            merged[key] = {
                **row,
                "count": 0,
                "ancestors": [] if row.get("ancestors") is not None else None,
            }
        merged[key]["count"] += row["count"]
        if row.get("ancestors") is not None:
            merged[key]["ancestors"].extend(row["ancestors"])
    return [merged[key] for key in sorted(merged)]


def _systematic_allocations(groups, budget: int, seed: int):
    with localcontext() as context:
        context.prec = 100
        probabilities = [
            Decimal(row["count"]) * row["individual_weight"] for row in groups
        ]
        total = sum(probabilities, Decimal(0))
        if total <= 0:
            raise RuntimeError("cannot resample a zero-mass particle population")
        probabilities = [value / total for value in probabilities]
        rng = random.Random(seed)
        offset = Decimal(str(rng.random())) / Decimal(budget)
        positions = [offset + Decimal(index) / Decimal(budget) for index in range(budget)]
        counts = [0 for _ in groups]
        cursor = 0
        cumulative = probabilities[0]
        for position in positions:
            while cursor < len(groups) - 1 and position >= cumulative:
                cursor += 1
                cumulative += probabilities[cursor]
            counts[cursor] += 1
    return counts, str(offset)


def _selected_ancestors(row, count: int, seed: int):
    ancestors = row.get("ancestors")
    if ancestors is None:
        return None
    if not ancestors or count == 0:
        return []
    rng = random.Random(seed)
    start = rng.randrange(len(ancestors))
    return [ancestors[(start + index) % len(ancestors)] for index in range(count)]


def systematic_resample(groups, budget: int, seed: int, track_ancestry: bool):
    allocations, offset = _systematic_allocations(groups, budget, seed)
    resampled = []
    with localcontext() as context:
        context.prec = 100
        unit_weight = Decimal(1) / Decimal(budget)
        for ordinal, (row, count) in enumerate(zip(groups, allocations, strict=True)):
            if not count:
                continue
            resampled.append({
                "configuration_key": row["configuration_key"],
                "world": row["world"],
                "queue": row["queue"],
                "count": count,
                "individual_weight": unit_weight,
                "ancestors": _selected_ancestors(
                    row, count, stream_seed(seed, "ancestor", ordinal)
                ) if track_ancestry else None,
            })
    return _merge_groups(resampled), allocations, offset


def particle_filter_episode(
    program,
    entities,
    initial_world,
    actions,
    evidence,
    budget: int,
    base_seed: int,
    stream_parts: Sequence[Any],
    ess_threshold_fraction: float = 0.5,
    track_ancestry: bool = False,
):
    if len(actions) != len(evidence):
        raise ValueError("V52 particle actions and evidence lengths differ")
    initial_key = _configuration_key(initial_world, [])
    with localcontext() as context:
        context.prec = 100
        unit_weight = Decimal(1) / Decimal(budget)
    groups = [{
        "configuration_key": initial_key,
        "world": dict(initial_world),
        "queue": [],
        "count": budget,
        "individual_weight": unit_weight,
        "ancestors": list(range(budget)) if track_ancestry else None,
    }]
    log_likelihood = Decimal(0)
    diagnostics = {
        "ticks": [],
        "resampling_stream_ids": [],
        "resampling_fingerprints": [],
        "transition_groups": 0,
        "extinct": False,
    }
    with localcontext() as context:
        context.prec = 100
        for tick, (action, observed) in enumerate(zip(actions, evidence, strict=True)):
            candidates = []
            for group in groups:
                branches = _unit_transition(
                    program, entities, group["world"], group["queue"], action, tick
                )
                diagnostics["transition_groups"] += 1
                for branch in branches.values():
                    if not observation_matches(branch["world"], observed):
                        continue
                    branch_probability = (
                        Decimal(branch["mass"].numerator) / Decimal(branch["mass"].denominator)
                    )
                    candidates.append({
                        "configuration_key": _configuration_key(branch["world"], branch["queue"]),
                        "world": dict(branch["world"]),
                        "queue": list(branch["queue"]),
                        "count": group["count"],
                        "individual_weight": group["individual_weight"] * branch_probability,
                        "ancestors": list(group["ancestors"]) if track_ancestry else None,
                    })
            candidates = _merge_groups(candidates)
            increment = sum(
                Decimal(row["count"]) * row["individual_weight"] for row in candidates
            )
            if increment <= 0:
                diagnostics["extinct"] = True
                return None, [], diagnostics
            log_likelihood += increment.ln()
            for row in candidates:
                row["individual_weight"] /= increment
            particle_count = sum(row["count"] for row in candidates)
            squared = sum(
                Decimal(row["count"]) * row["individual_weight"] ** 2
                for row in candidates
            )
            ess = Decimal(1) / squared
            must_resample = (
                particle_count > budget
                or ess < Decimal(str(ess_threshold_fraction)) * Decimal(budget)
            )
            resampled = False
            allocations = []
            resampling_offset = None
            if must_resample:
                parts = (*stream_parts, tick)
                seed = stream_seed(base_seed, *parts)
                identifier = stream_id(base_seed, *parts)
                candidates, allocations, resampling_offset = systematic_resample(
                    candidates, budget, seed, track_ancestry
                )
                diagnostics["resampling_stream_ids"].append(identifier)
                diagnostics["resampling_fingerprints"].append(
                    sha256_text(canonical_json({
                        "systematic_offset": resampling_offset,
                        "allocations": allocations,
                        "states": [row["configuration_key"] for row in candidates],
                    }))
                )
                resampled = True
            groups = candidates
            state = configuration_distribution(groups)
            ancestors = {
                ancestor
                for row in groups for ancestor in (row.get("ancestors") or [])
            }
            diagnostics["ticks"].append({
                "tick": tick,
                "likelihood_increment": float(increment),
                "particle_count_before_resampling": particle_count,
                "weighted_group_count": len(groups),
                "ess": float(ess),
                "ess_fraction": float(ess / Decimal(max(1, particle_count))),
                "resampled": resampled,
                "distinct_configurations": len(state),
                "distinct_ancestors": len(ancestors) if track_ancestry else None,
            })
    return log_likelihood, groups, diagnostics


def configuration_distribution(groups):
    result = {}
    with localcontext() as context:
        context.prec = 100
        for row in groups:
            mass = Decimal(row["count"]) * row["individual_weight"]
            result[row["configuration_key"]] = result.get(
                row["configuration_key"], Decimal(0)
            ) + mass
        total = sum(result.values(), Decimal(0))
        if total:
            result = {key: value / total for key, value in result.items()}
    return result


def _normalize_log_weights(log_weights):
    with localcontext() as context:
        context.prec = 100
        valid = [value for value in log_weights if value is not None]
        if not valid:
            raise RuntimeError("all V52 static hypotheses are extinct")
        maximum = max(valid)
        raw = [Decimal(0) if value is None else (value - maximum).exp() for value in log_weights]
        total = sum(raw, Decimal(0))
        return [value / total for value in raw]


def _logsumexp(log_weights):
    with localcontext() as context:
        context.prec = 100
        valid = [value for value in log_weights if value is not None]
        if not valid:
            return None
        maximum = max(valid)
        return maximum + sum((value - maximum).exp() for value in valid).ln()


def probability_marginal(registry, program_weights):
    result = {value: Decimal(0) for value in ("1/4", "1/2", "3/4")}
    for mechanic, weight in zip(registry, program_weights, strict=True):
        result[mechanic["probability"]] += weight
    return result


def particle_inference(
    registry,
    supports,
    query,
    budget: int,
    base_seed: int,
    population: str,
    record_id: str,
    repeat: int,
    ess_threshold_fraction: float = 0.5,
    likelihood_power: int = 1,
    track_ancestry: bool = False,
):
    support_logs, support_diagnostics = [], []
    for program_index, mechanic in enumerate(registry):
        total = Decimal(0)
        program_diagnostics = []
        possible = True
        for episode, support in enumerate(supports):
            world = {row["atom"]: row["allowed_values"][0] for row in support["initial_state"]}
            log_likelihood, _, diagnostics = particle_filter_episode(
                mechanic["program"], support["entities"], world,
                support["actions"], support["observations"], budget, base_seed,
                (population, record_id, program_index, budget, repeat, f"support-{episode}"),
                ess_threshold_fraction, track_ancestry,
            )
            program_diagnostics.append(diagnostics)
            if log_likelihood is None:
                possible = False
                break
            total += Decimal(likelihood_power) * log_likelihood
        support_logs.append(total if possible else None)
        support_diagnostics.append(program_diagnostics)
    support_program = _normalize_log_weights(support_logs)
    support_log_evidence = _logsumexp(support_logs) - Decimal(len(registry)).ln()

    query_logs, filters, query_diagnostics = [], [], []
    prefix = query["prefix_length"]
    for program_index, mechanic in enumerate(registry):
        if not support_program[program_index]:
            query_logs.append(None)
            filters.append([])
            query_diagnostics.append({"extinct": True, "ticks": []})
            continue
        world = {row["atom"]: row["allowed_values"][0] for row in query["initial_state"]}
        log_likelihood, groups, diagnostics = particle_filter_episode(
            mechanic["program"], query["entities"], world,
            query["actions"][:prefix], query["observations"], budget, base_seed,
            (population, record_id, program_index, budget, repeat, "query"),
            ess_threshold_fraction, track_ancestry,
        )
        filters.append(groups)
        query_diagnostics.append(diagnostics)
        query_logs.append(
            None if log_likelihood is None
            else support_program[program_index].ln() + Decimal(likelihood_power) * log_likelihood
        )
    query_program = _normalize_log_weights(query_logs)
    query_conditional_log_evidence = _logsumexp(query_logs)
    joint, metadata, configuration = {}, {}, {}
    for program_index, (mechanic, groups) in enumerate(zip(registry, filters, strict=True)):
        for configuration_key, state_mass in configuration_distribution(groups).items():
            key = canonical_json({
                "program": mechanic["key"], "configuration": configuration_key
            })
            mass = query_program[program_index] * state_mass
            joint[key] = mass
            configuration[configuration_key] = configuration.get(
                configuration_key, Decimal(0)
            ) + mass
            metadata[key] = {
                "program_index": program_index,
                "program_key": mechanic["key"],
                "program_ordinal": mechanic["program_ordinal"],
                "probability_ordinal": mechanic["probability_ordinal"],
                "configuration_key": configuration_key,
            }
    suffix = particle_suffix_predictive(registry, query_program, query, filters)
    return {
        "support_program": support_program,
        "query_program": query_program,
        "probability": probability_marginal(registry, query_program),
        "joint": joint,
        "metadata": metadata,
        "configuration": configuration,
        "suffix": suffix,
        "support_log_evidence_by_program": support_logs,
        "query_log_weight_by_program": query_logs,
        "record_log_evidence": support_log_evidence + query_conditional_log_evidence,
        "support_diagnostics": support_diagnostics,
        "query_diagnostics": query_diagnostics,
    }


def particle_suffix_predictive(registry, program_weights, query, filters):
    prefix = query["prefix_length"]
    result = {}
    with localcontext() as context:
        context.prec = 100
        for mechanic, program_weight, groups in zip(
            registry, program_weights, filters, strict=True
        ):
            if not program_weight:
                continue
            for group in groups:
                state_mass = Decimal(group["count"]) * group["individual_weight"]
                seed = {
                    _configuration_key_with_history(group["world"], group["queue"], []): {
                        "world": dict(group["world"]), "queue": list(group["queue"]),
                        "history": [], "mass": Fraction(1),
                    }
                }
                continuations = advance_configurations(
                    mechanic["program"], query["entities"], seed,
                    query["actions"][prefix:], prefix,
                )
                for continuation in continuations.values():
                    key = canonical_json(continuation["history"])
                    probability = (
                        program_weight * state_mass
                        * Decimal(continuation["mass"].numerator)
                        / Decimal(continuation["mass"].denominator)
                    )
                    result[key] = result.get(key, Decimal(0)) + probability
        total = sum(result.values(), Decimal(0))
        return {key: value / total for key, value in result.items()} if total else {}
