"""Continuous-parameter exact, SMC-squared, and PMMH utilities for V53."""
from __future__ import annotations

import copy
import json
import math
import random
from decimal import Decimal, localcontext
from fractions import Fraction
from itertools import product
from typing import Any, Sequence

import numpy as np

from v22_relational import canonical_json, sha256_text
from v42_stateful import effect, relation, unary, world_signature
from v46_stochastic import (
    _apply_payload,
    _condition_holds,
    _configuration_key,
    _deliver,
    _rule,
    _validate_action,
    canonical_program,
    delayed,
    stochastic,
)
from v46_stochastic import mechanic_registry as v46_registry
from v47_sampling import mechanic_registry as v47_registry
from v48_composition import mechanic_registry as v48_registry
from v49_belief import _configuration_key_with_history, advance_configurations
from v49_belief import mechanic_registry as v49_registry
from v50_belief import mechanic_registry as v50_registry
from v51_sbc import mechanic_registry as v51_registry
from v52_particle import (
    V52_FAMILIES,
    _merge_groups,
    _v52_candidates,
    configuration_distribution,
    mechanic_registry as v52_registry,
    observation_matches,
    stream_id,
    stream_seed,
    systematic_resample,
)


THETA_TOKEN = "$theta"
def parameterize_program(program):
    result = copy.deepcopy(program)
    replaced = 0
    for rule in result["rules"]:
        for branch in rule["stochastic_immediate"]:
            branch["probability"] = THETA_TOKEN
            replaced += 1
        for delayed in rule["stochastic_delayed"]:
            delayed["probability"] = THETA_TOKEN
            replaced += 1
    if replaced != 1:
        raise ValueError("V53 templates require exactly one stochastic probability")
    return result


def template_key(program):
    return canonical_json(parameterize_program(program))


def theta_text(theta: float) -> str:
    value = f"{theta:.17g}".rstrip("0").rstrip(".")
    return value if value else "0"


def instantiate_program(template, theta: float):
    if not 0 < theta < 1:
        raise ValueError("theta must be strictly between zero and one")
    result = copy.deepcopy(template)
    replaced = 0
    for rule in result["rules"]:
        for branch in rule["stochastic_immediate"]:
            if branch["probability"] == THETA_TOKEN:
                branch["probability"] = theta_text(theta)
                replaced += 1
        for delayed in rule["stochastic_delayed"]:
            if delayed["probability"] == THETA_TOKEN:
                delayed["probability"] = theta_text(theta)
                replaced += 1
    if replaced != 1:
        raise ValueError("V53 template did not contain one theta token")
    probability = next(
        branch["probability"]
        for rule in result["rules"]
        for branch in (*rule["stochastic_immediate"], *rule["stochastic_delayed"])
    )
    if not 0 < Fraction(probability) < 1:
        raise ValueError("V53 continuous probability must lie strictly inside (0, 1)")
    # The template was canonicalized before its probability became continuous.
    # Do not call V46's frozen finite-vocabulary validator here.
    return result


def _v53_candidates():
    """Extend the prior finite pool without reusing a parameterized template.

    The cross-action pool through V52 used only set-true/toggle stochastic
    effects.  Adding the symmetric set-false case expands that same family
    without changing its action topology or introducing a new mechanism.
    """
    yield from _v52_candidates()
    targets = [
        *(unary(predicate, variable) for predicate, variable in product(
            ("active", "marked", "ready"), ("actor", "target")
        )),
        relation("actor", "target"),
        relation("target", "actor"),
    ]
    for probability, delay_ticks, target in product(
        (Fraction(1, 4), Fraction(1, 2), Fraction(3, 4)), (1, 2), targets
    ):
        yield "cross_action_stochastic_composition", probability, canonical_program({"rules": [
            _rule("pulse", stochastic_delayed=[delayed(
                delay_ticks, stochastic(str(probability), effect("set_false", target))
            )]),
            _rule("route", deterministic_immediate=[effect("toggle", target)]),
        ]})


def mechanic_registry(template_seed: int = 5303):
    previous = {
        template_key(row["program"])
        for source in (
            v46_registry, v47_registry, v48_registry, v49_registry,
            v50_registry, v51_registry, v52_registry,
        )
        for row in source()
    }
    grouped = {family: {} for family in V52_FAMILIES}
    for family, _, program in _v53_candidates():
        key = template_key(program)
        if key in previous:
            continue
        grouped[family][key] = parameterize_program(program)
    rows = []
    for family in V52_FAMILIES:
        candidates = sorted(
            grouped[family].items(),
            key=lambda row: sha256_text(f"v53-template|{template_seed}|{row[0]}"),
        )
        if len(candidates) < 2:
            raise RuntimeError(f"V53 lacks fresh templates for {family}")
        for family_ordinal, (key, template) in enumerate(candidates[:2]):
            stochastic_rule = next(
                rule for rule in template["rules"]
                if rule["stochastic_immediate"] or rule["stochastic_delayed"]
            )
            delayed = stochastic_rule["stochastic_delayed"]
            rows.append({
                "family": family,
                "family_ordinal": family_ordinal,
                "timing": "delayed" if delayed else "immediate",
                "delay": delayed[0]["delay"] if delayed else 0,
                "template": template,
                "key": key,
                "id": f"template_{sha256_text(key)[:16]}",
            })
    rows.sort(key=lambda row: row["key"])
    for index, row in enumerate(rows):
        row["program_ordinal"] = index
    if len(rows) != 8 or len({row["key"] for row in rows}) != 8:
        raise RuntimeError("V53 registry must contain eight fresh unique templates")
    return rows


def scaled_beta_sample(seed: int, low=0.05, high=0.95, alpha=2.0, beta=2.0):
    value = random.Random(seed).betavariate(alpha, beta)
    return low + (high - low) * value


def scaled_beta_log_pdf(theta, low=0.05, high=0.95, alpha=2.0, beta=2.0):
    if not low < theta < high:
        return -math.inf
    x = (theta - low) / (high - low)
    log_beta = math.lgamma(alpha) + math.lgamma(beta) - math.lgamma(alpha + beta)
    return (
        (alpha - 1) * math.log(x)
        + (beta - 1) * math.log1p(-x)
        - log_beta
        - math.log(high - low)
    )


def theta_to_logit(theta, low=0.05, high=0.95):
    x = (theta - low) / (high - low)
    return math.log(x) - math.log1p(-x)


def logit_to_theta(value, low=0.05, high=0.95):
    if value >= 0:
        x = 1 / (1 + math.exp(-value))
    else:
        exp_value = math.exp(value)
        x = exp_value / (1 + exp_value)
    return low + (high - low) * x


def log_abs_theta_jacobian(theta, low=0.05, high=0.95):
    x = (theta - low) / (high - low)
    return math.log(high - low) + math.log(x) + math.log1p(-x)


def quadrature_rule(nodes: int, parameter_model):
    raw_nodes, raw_weights = np.polynomial.legendre.leggauss(nodes)
    low, high = parameter_model["support"]
    theta = low + (raw_nodes + 1) * (high - low) / 2
    weights = raw_weights * (high - low) / 2
    prior = np.array([
        math.exp(scaled_beta_log_pdf(
            float(value), low, high,
            parameter_model["alpha"], parameter_model["beta"],
        ))
        for value in theta
    ])
    weights = weights * prior
    weights = weights / weights.sum()
    return [(float(value), float(weight)) for value, weight in zip(theta, weights)]


def _initial_world(episode):
    return {
        row["atom"]: row["allowed_values"][0]
        for row in episode["initial_state"]
    }


def record_episodes(record):
    episodes = list(record["supports"])
    query = record["query"]
    episodes.append({
        **query,
        "actions": query["actions"][: query["prefix_length"]],
    })
    return episodes


def continuous_advance_configurations(
    program, entities, configurations, actions, start_tick: int
):
    """V49 transition semantics with an arbitrary rational branch probability."""
    rules = {row["action"]: row for row in program["rules"]}
    current = configurations
    for offset, action in enumerate(actions):
        tick = start_tick + offset
        action_id, binding = _validate_action(action, entities)
        successors = {}
        for configuration in current.values():
            world, queue = _deliver(
                configuration["queue"], tick, configuration["world"], entities
            )
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
                    probability = Fraction(branch["probability"])
                    alternatives = [(base, queue, mass * (1 - probability))]
                    if rule["stochastic_immediate"]:
                        alternatives.append((
                            _apply_payload(branch["effect"], base, binding, entities),
                            queue,
                            mass * probability,
                        ))
                    else:
                        event = {
                            "due": tick + branch["delay"],
                            "effect": branch["effect"],
                            "binding": dict(binding),
                        }
                        alternatives.append((base, [*queue, event], mass * probability))
                else:
                    alternatives = [(base, queue, mass)]
            for next_world, next_queue, next_mass in alternatives:
                if not next_mass:
                    continue
                history = [*configuration["history"], world_signature(next_world)]
                key = _configuration_key_with_history(
                    next_world, next_queue, history
                )
                if key not in successors:
                    successors[key] = {
                        "world": dict(next_world),
                        "queue": list(next_queue),
                        "history": history,
                        "mass": Fraction(0),
                    }
                successors[key]["mass"] += next_mass
        current = successors
    return current


def continuous_sequential_filter(program, entities, initial_world, actions, evidence):
    if len(actions) != len(evidence):
        raise ValueError("V53 exact actions and evidence lengths differ")
    current = {
        _configuration_key_with_history(initial_world, [], []): {
            "world": dict(initial_world),
            "queue": [],
            "history": [],
            "mass": Fraction(1),
        }
    }
    for tick, (action, observed) in enumerate(zip(actions, evidence, strict=True)):
        current = continuous_advance_configurations(
            program, entities, current, [action], tick
        )
        current = {
            key: row for key, row in current.items()
            if observation_matches(row["world"], observed)
        }
        if not current:
            return Fraction(0), {}
    return sum((row["mass"] for row in current.values()), Fraction(0)), current


def continuous_unit_transition(program, entities, world, queue, action, tick):
    seed = {
        _configuration_key_with_history(world, queue, []): {
            "world": dict(world),
            "queue": list(queue),
            "history": [],
            "mass": Fraction(1),
        }
    }
    return continuous_advance_configurations(
        program, entities, seed, [action], tick
    )


def continuous_particle_filter_episode(
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
    """V52 Rao--Blackwellized filter with V53 continuous transitions."""
    if len(actions) != len(evidence):
        raise ValueError("V53 particle actions and evidence lengths differ")
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
                branches = continuous_unit_transition(
                    program, entities, group["world"], group["queue"], action, tick
                )
                diagnostics["transition_groups"] += 1
                for branch in branches.values():
                    if not observation_matches(branch["world"], observed):
                        continue
                    branch_probability = (
                        Decimal(branch["mass"].numerator)
                        / Decimal(branch["mass"].denominator)
                    )
                    candidates.append({
                        "configuration_key": _configuration_key(
                            branch["world"], branch["queue"]
                        ),
                        "world": dict(branch["world"]),
                        "queue": list(branch["queue"]),
                        "count": group["count"],
                        "individual_weight": (
                            group["individual_weight"] * branch_probability
                        ),
                        "ancestors": (
                            list(group["ancestors"]) if track_ancestry else None
                        ),
                    })
            candidates = _merge_groups(candidates)
            increment = sum(
                Decimal(row["count"]) * row["individual_weight"]
                for row in candidates
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
            allocations, resampling_offset, resampled = [], None, False
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
                "distinct_configurations": len(configuration_distribution(groups)),
                "distinct_ancestors": len(ancestors) if track_ancestry else None,
            })
    return log_likelihood, groups, diagnostics


def exact_theta_path(template_row, theta, record):
    program = instantiate_program(template_row["template"], theta)
    log_likelihood = 0.0
    query_configurations = None
    query_likelihood = None
    episodes = record_episodes(record)
    for ordinal, episode in enumerate(episodes):
        likelihood, configurations = continuous_sequential_filter(
            program, episode["entities"], _initial_world(episode),
            episode["actions"], episode["observations"],
        )
        if not likelihood:
            return None
        log_likelihood += math.log(likelihood.numerator) - math.log(likelihood.denominator)
        if ordinal == len(episodes) - 1:
            query_configurations = configurations
            query_likelihood = likelihood
    configuration = {}
    for row in query_configurations.values():
        key = canonical_json({
            "world": sorted(row["world"].items()),
            "queue": sorted(row["queue"], key=canonical_json),
        })
        probability = float(row["mass"] / query_likelihood)
        configuration[key] = configuration.get(key, 0.0) + probability
    suffix = {}
    query = record["query"]
    prefix = query["prefix_length"]
    for row in query_configurations.values():
        history_weight = float(row["mass"] / query_likelihood)
        seed = {
            _configuration_key_with_history(row["world"], row["queue"], []): {
                "world": dict(row["world"]), "queue": list(row["queue"]),
                "history": [], "mass": Fraction(1),
            }
        }
        continuations = continuous_advance_configurations(
            program, query["entities"], seed, query["actions"][prefix:], prefix
        )
        for continuation in continuations.values():
            key = canonical_json(continuation["history"])
            suffix[key] = suffix.get(key, 0.0) + history_weight * float(
                continuation["mass"]
            )
    return {
        "log_likelihood": log_likelihood,
        "configuration": normalize_float_map(configuration),
        "suffix": normalize_float_map(suffix),
    }


def logsumexp(values):
    valid = [value for value in values if math.isfinite(value)]
    if not valid:
        return -math.inf
    maximum = max(valid)
    return maximum + math.log(sum(math.exp(value - maximum) for value in valid))


def normalize_log_weights(values):
    normalizer = logsumexp(values)
    if not math.isfinite(normalizer):
        raise RuntimeError("all continuous-parameter hypotheses have zero mass")
    return [0.0 if not math.isfinite(value) else math.exp(value - normalizer) for value in values]


def normalize_float_map(values):
    total = sum(values.values())
    if total <= 0:
        return {}
    return {key: value / total for key, value in values.items()}


def theta_bin(theta, parameter_model, bins):
    low, high = parameter_model["support"]
    return min(bins - 1, max(0, int((theta - low) / (high - low) * bins)))


def exact_inference(registry, record, config):
    rule = quadrature_rule(config["exactBenchmark"]["quadratureNodes"], config["parameterModel"])
    rows, log_masses = [], []
    for program_index, template in enumerate(registry):
        for node_index, (theta, prior_weight) in enumerate(rule):
            result = exact_theta_path(template, theta, record)
            log_mass = (
                -math.log(len(registry)) + math.log(prior_weight)
                + (result["log_likelihood"] if result else -math.inf)
            )
            rows.append({
                "program_index": program_index,
                "node_index": node_index,
                "theta": theta,
                "result": result,
            })
            log_masses.append(log_mass)
    weights = normalize_log_weights(log_masses)
    program = [0.0 for _ in registry]
    theta_values, theta_weights = [], []
    joint_bins, configuration, suffix, atoms = {}, {}, {}, []
    for row, weight in zip(rows, weights, strict=True):
        if not weight:
            continue
        program[row["program_index"]] += weight
        theta_values.append(row["theta"])
        theta_weights.append(weight)
        bin_key = (
            f"{row['program_index']}:"
            f"{theta_bin(row['theta'], config['parameterModel'], config['exactBenchmark']['thetaBins'])}"
        )
        joint_bins[bin_key] = joint_bins.get(bin_key, 0.0) + weight
        for key, probability in row["result"]["configuration"].items():
            configuration[key] = configuration.get(key, 0.0) + weight * probability
            atoms.append({
                "program_index": row["program_index"],
                "theta": row["theta"],
                "configuration_key": key,
                "weight": weight * probability,
            })
        for key, probability in row["result"]["suffix"].items():
            suffix[key] = suffix.get(key, 0.0) + weight * probability
    atom_total = sum(row["weight"] for row in atoms)
    for atom in atoms:
        atom["weight"] /= atom_total
    return {
        "program": normalize_float_sequence(program),
        "theta_values": theta_values,
        "theta_weights": normalize_float_sequence(theta_weights),
        "joint_bins": normalize_float_map(joint_bins),
        "configuration": normalize_float_map(configuration),
        "suffix": normalize_float_map(suffix),
        "log_evidence": logsumexp(log_masses),
        "rows": rows,
        "weights": weights,
        "atoms": atoms,
    }


def exact_conditional_theta(exact, program_index):
    values, weights = [], []
    for row, weight in zip(exact["rows"], exact["weights"], strict=True):
        if row["program_index"] == program_index and weight:
            values.append(row["theta"])
            weights.append(weight)
    return values, normalize_float_sequence(weights)


def normalize_float_sequence(values):
    total = sum(values)
    if total <= 0:
        return [0.0 for _ in values]
    return [value / total for value in values]


def _episode_particle_path(
    template,
    theta,
    episode,
    inner_budget,
    inner_seed,
    stream_parts,
    inner_ess,
    track_ancestry=False,
):
    return continuous_particle_filter_episode(
        instantiate_program(template, theta),
        episode["entities"], _initial_world(episode), episode["actions"],
        episode["observations"], inner_budget, inner_seed, stream_parts,
        inner_ess, track_ancestry,
    )


def _full_particle_likelihood(
    template,
    theta,
    episodes,
    inner_budget,
    inner_seed,
    stream_prefix,
    inner_ess,
    purpose,
    track_ancestry=False,
):
    total = Decimal(0)
    diagnostics, last_groups = [], []
    for episode_index, episode in enumerate(episodes):
        log_likelihood, groups, diagnostic = _episode_particle_path(
            template, theta, episode, inner_budget, inner_seed,
            (*stream_prefix, episode_index, purpose), inner_ess, track_ancestry,
        )
        diagnostics.append(diagnostic)
        if log_likelihood is None:
            return None, [], diagnostics
        total += log_likelihood
        last_groups = groups
    return float(total), last_groups, diagnostics


def _systematic_indices(weights, count, seed):
    rng = random.Random(seed)
    offset = rng.random() / count
    positions = [offset + index / count for index in range(count)]
    indices, cursor, cumulative = [], 0, weights[0]
    for position in positions:
        while cursor < len(weights) - 1 and position >= cumulative:
            cursor += 1
            cumulative += weights[cursor]
        indices.append(cursor)
    return indices, offset


def _diagnostic_streams(diagnostics):
    ids, fingerprints = [], []
    for diagnostic in diagnostics:
        ids.extend(diagnostic.get("resampling_stream_ids", []))
        fingerprints.extend(diagnostic.get("resampling_fingerprints", []))
    return ids, fingerprints


def _outer_program_smc(
    template_row,
    program_index,
    record,
    config,
    outer_budget,
    repeat,
    population,
    disable_outer_resampling=False,
    likelihood_power=1,
    track_ancestry=False,
):
    specification = config["smcSquared"]
    inner_budget = specification["innerStateParticleBudget"]
    particles = []
    for particle_index in range(outer_budget):
        theta = scaled_beta_sample(stream_seed(
            config["population"]["outerParticleSeed"],
            population, record["id"], program_index, outer_budget, repeat,
            particle_index, "prior",
        ))
        particles.append({
            "theta": theta,
            "weight": 1 / outer_budget,
            "log_likelihood": 0.0,
            "ancestor": particle_index,
            "last_groups": [],
        })
    log_evidence = 0.0
    episodes = record_episodes(record)
    diagnostics = {
        "outer_resampling_stream_ids": [],
        "outer_resampling_fingerprints": [],
        "inner_resampling_stream_ids": [],
        "inner_resampling_fingerprints": [],
        "outer_ess_fractions": [],
        "inner_ess_fractions": [],
        "move_attempts": 0,
        "move_accepts": 0,
    }
    outer_resampling_count = 0
    for episode_index, episode in enumerate(episodes):
        log_unnormalized = []
        for slot, particle in enumerate(particles):
            if particle["weight"] <= 0:
                particle["last_groups"] = []
                log_unnormalized.append(-math.inf)
                continue
            log_likelihood, groups, diagnostic = _episode_particle_path(
                template_row["template"], particle["theta"], episode,
                inner_budget, config["population"]["innerParticleSeed"],
                (
                    population, record["id"], program_index, outer_budget, repeat,
                    slot, episode_index, "update",
                ),
                specification["innerEssThresholdFraction"], track_ancestry,
            )
            ids, fingerprints = _diagnostic_streams([diagnostic])
            diagnostics["inner_resampling_stream_ids"].extend(ids)
            diagnostics["inner_resampling_fingerprints"].extend(fingerprints)
            diagnostics["inner_ess_fractions"].extend(
                tick["ess_fraction"] for tick in diagnostic.get("ticks", [])
            )
            if log_likelihood is None:
                log_unnormalized.append(-math.inf)
                particle["last_groups"] = []
                continue
            increment = likelihood_power * float(log_likelihood)
            particle["log_likelihood"] += increment
            particle["last_groups"] = groups
            log_unnormalized.append(math.log(particle["weight"]) + increment)
        increment_log_evidence = logsumexp(log_unnormalized)
        if not math.isfinite(increment_log_evidence):
            return None
        log_evidence += increment_log_evidence
        weights = normalize_log_weights(log_unnormalized)
        for particle, weight in zip(particles, weights, strict=True):
            particle["weight"] = weight
        ess = 1 / sum(weight * weight for weight in weights)
        diagnostics["outer_ess_fractions"].append(ess / outer_budget)
        if (
            not disable_outer_resampling
            and ess < specification["outerEssThresholdFraction"] * outer_budget
        ):
            outer_stream_parts = (
                population, record["id"], program_index, outer_budget, repeat,
                episode_index, "outer-resample", outer_resampling_count,
            )
            seed = stream_seed(config["population"]["outerParticleSeed"], *outer_stream_parts)
            identifier = stream_id(config["population"]["outerParticleSeed"], *outer_stream_parts)
            indices, offset = _systematic_indices(weights, outer_budget, seed)
            particles = [copy.deepcopy(particles[index]) for index in indices]
            for slot, particle in enumerate(particles):
                particle["weight"] = 1 / outer_budget
                for move in range(specification["rejuvenationStepsPerOuterResampling"]):
                    proposal_seed = stream_seed(
                        config["population"]["rejuvenationSeed"],
                        population, record["id"], program_index, outer_budget, repeat,
                        slot, episode_index, outer_resampling_count, move, "proposal",
                    )
                    rng = random.Random(proposal_seed)
                    current_theta = particle["theta"]
                    proposal_theta = logit_to_theta(
                        theta_to_logit(current_theta)
                        + rng.gauss(0, specification["proposalStandardDeviation"])
                    )
                    proposed_ll, proposed_groups, proposed_diagnostics = _full_particle_likelihood(
                        template_row["template"], proposal_theta,
                        episodes[: episode_index + 1], inner_budget,
                        config["population"]["innerParticleSeed"],
                        (
                            population, record["id"], program_index, outer_budget,
                            repeat, slot, episode_index, outer_resampling_count, move,
                        ),
                        specification["innerEssThresholdFraction"], "move",
                        track_ancestry,
                    )
                    ids, fingerprints = _diagnostic_streams(proposed_diagnostics)
                    diagnostics["inner_resampling_stream_ids"].extend(ids)
                    diagnostics["inner_resampling_fingerprints"].extend(fingerprints)
                    diagnostics["inner_ess_fractions"].extend(
                        tick["ess_fraction"]
                        for diagnostic in proposed_diagnostics
                        for tick in diagnostic.get("ticks", [])
                    )
                    diagnostics["move_attempts"] += 1
                    if proposed_ll is None:
                        continue
                    proposed_ll *= likelihood_power
                    current_log_target = (
                        scaled_beta_log_pdf(current_theta)
                        + particle["log_likelihood"]
                        + log_abs_theta_jacobian(current_theta)
                    )
                    proposed_log_target = (
                        scaled_beta_log_pdf(proposal_theta)
                        + proposed_ll
                        + log_abs_theta_jacobian(proposal_theta)
                    )
                    accept_seed = stream_seed(
                        config["population"]["rejuvenationSeed"],
                        population, record["id"], program_index, outer_budget, repeat,
                        slot, episode_index, outer_resampling_count, move, "accept",
                    )
                    if math.log(max(random.Random(accept_seed).random(), 1e-300)) < min(
                        0.0, proposed_log_target - current_log_target
                    ):
                        particle.update({
                            "theta": proposal_theta,
                            "log_likelihood": proposed_ll,
                            "last_groups": proposed_groups,
                        })
                        diagnostics["move_accepts"] += 1
            diagnostics["outer_resampling_stream_ids"].append(identifier)
            diagnostics["outer_resampling_fingerprints"].append(sha256_text(canonical_json({
                "offset": offset,
                "indices": indices,
                "theta": [theta_text(row["theta"]) for row in particles],
            })))
            outer_resampling_count += 1
    return {
        "log_evidence": log_evidence,
        "particles": particles,
        "diagnostics": diagnostics,
    }


def smc2_inference(
    registry,
    record,
    config,
    outer_budget,
    repeat,
    population,
    disable_outer_resampling=False,
    likelihood_power=1,
    track_ancestry=False,
):
    programs = []
    for program_index, template in enumerate(registry):
        result = _outer_program_smc(
            template, program_index, record, config, outer_budget, repeat, population,
            disable_outer_resampling, likelihood_power, track_ancestry,
        )
        programs.append(result)
    log_program = [
        -math.inf if result is None else result["log_evidence"] - math.log(len(registry))
        for result in programs
    ]
    program_weights = normalize_log_weights(log_program)
    theta_values, theta_weights = [], []
    joint_bins, configuration, suffix, atoms = {}, {}, {}, []
    query = record["query"]
    prefix = query["prefix_length"]
    for program_index, (template, program_result, program_weight) in enumerate(
        zip(registry, programs, program_weights, strict=True)
    ):
        if program_result is None or not program_weight:
            continue
        for particle in program_result["particles"]:
            mass = program_weight * particle["weight"]
            theta_values.append(particle["theta"])
            theta_weights.append(mass)
            bin_key = (
                f"{program_index}:"
                f"{theta_bin(particle['theta'], config['parameterModel'], config['exactBenchmark']['thetaBins'])}"
            )
            joint_bins[bin_key] = joint_bins.get(bin_key, 0.0) + mass
            state = configuration_distribution(particle["last_groups"])
            for key, probability in state.items():
                atom_mass = mass * float(probability)
                configuration[key] = configuration.get(key, 0.0) + atom_mass
                atoms.append({
                    "program_index": program_index,
                    "theta": particle["theta"],
                    "configuration_key": key,
                    "weight": atom_mass,
                })
            for group in particle["last_groups"]:
                state_mass = float(Decimal(group["count"]) * group["individual_weight"])
                seed = {
                    _configuration_key_with_history(group["world"], group["queue"], []): {
                        "world": dict(group["world"]), "queue": list(group["queue"]),
                        "history": [], "mass": Fraction(1),
                    }
                }
                continuations = continuous_advance_configurations(
                    instantiate_program(template["template"], particle["theta"]),
                    query["entities"], seed, query["actions"][prefix:], prefix,
                )
                for continuation in continuations.values():
                    key = canonical_json(continuation["history"])
                    suffix[key] = suffix.get(key, 0.0) + (
                        mass * state_mass * float(continuation["mass"])
                    )
    atom_total = sum(row["weight"] for row in atoms)
    for atom in atoms:
        atom["weight"] /= atom_total
    return {
        "program": normalize_float_sequence(program_weights),
        "theta_values": theta_values,
        "theta_weights": normalize_float_sequence(theta_weights),
        "joint_bins": normalize_float_map(joint_bins),
        "configuration": normalize_float_map(configuration),
        "suffix": normalize_float_map(suffix),
        "log_evidence": logsumexp(log_program),
        "program_results": programs,
        "atoms": atoms,
    }


def pool_smc2_repeats(results):
    """Equal-weight posterior mixture of preregistered independent repeats."""
    if not results:
        raise ValueError("V53 repeat pool cannot be empty")
    count = len(results)

    def average_sequence(key):
        return normalize_float_sequence([
            sum(result[key][index] for result in results) / count
            for index in range(len(results[0][key]))
        ])

    def average_map(key):
        combined = {}
        for result in results:
            for item, value in result[key].items():
                combined[item] = combined.get(item, 0.0) + value / count
        return normalize_float_map(combined)

    theta_values, theta_weights, atoms = [], [], []
    for result in results:
        theta_values.extend(result["theta_values"])
        theta_weights.extend(weight / count for weight in result["theta_weights"])
        atoms.extend({**atom, "weight": atom["weight"] / count} for atom in result["atoms"])
    return {
        "program": average_sequence("program"),
        "theta_values": theta_values,
        "theta_weights": normalize_float_sequence(theta_weights),
        "joint_bins": average_map("joint_bins"),
        "configuration": average_map("configuration"),
        "suffix": average_map("suffix"),
        "log_evidence": logsumexp([result["log_evidence"] for result in results])
        - math.log(count),
        "atoms": atoms,
        "repeat_results": results,
    }


def pmmh_conditional_chains(
    template_row,
    record,
    config,
    chains=None,
    warmup=None,
    retained=None,
    inner_budget=None,
    seed_shift=0,
):
    specification = config["pmcmcReference"]
    chains = specification["chains"] if chains is None else chains
    warmup = specification["warmupIterationsPerChain"] if warmup is None else warmup
    retained = specification["retainedIterationsPerChain"] if retained is None else retained
    inner_budget = specification["innerStateParticleBudget"] if inner_budget is None else inner_budget
    episodes = record_episodes(record)
    results = []
    for chain in range(chains):
        theta = scaled_beta_sample(stream_seed(
            config["population"]["thetaPriorSeed"] + seed_shift,
            "pmmh", record["id"], template_row["key"], chain, "initial",
        ))
        current_ll, _, _ = _full_particle_likelihood(
            template_row["template"], theta, episodes, inner_budget,
            config["population"]["innerParticleSeed"] + seed_shift,
            ("pmmh", record["id"], template_row["key"], chain, "initial"),
            config["smcSquared"]["innerEssThresholdFraction"], "current",
        )
        draws, accepts = [], 0
        total_iterations = warmup + retained
        for iteration in range(total_iterations):
            proposal_rng = random.Random(stream_seed(
                config["population"]["rejuvenationSeed"] + seed_shift,
                "pmmh", record["id"], template_row["key"], chain, iteration,
                "proposal",
            ))
            proposal = logit_to_theta(
                theta_to_logit(theta)
                + proposal_rng.gauss(0, specification["proposalStandardDeviation"])
            )
            proposed_ll, _, _ = _full_particle_likelihood(
                template_row["template"], proposal, episodes, inner_budget,
                config["population"]["innerParticleSeed"] + seed_shift,
                ("pmmh", record["id"], template_row["key"], chain, iteration),
                config["smcSquared"]["innerEssThresholdFraction"], "proposal",
            )
            if proposed_ll is not None:
                current_target = (
                    scaled_beta_log_pdf(theta) + current_ll + log_abs_theta_jacobian(theta)
                )
                proposed_target = (
                    scaled_beta_log_pdf(proposal) + proposed_ll
                    + log_abs_theta_jacobian(proposal)
                )
                accept_draw = random.Random(stream_seed(
                    config["population"]["rejuvenationSeed"] + seed_shift,
                    "pmmh", record["id"], template_row["key"], chain, iteration,
                    "accept",
                )).random()
                if math.log(max(accept_draw, 1e-300)) < min(
                    0.0, proposed_target - current_target
                ):
                    theta, current_ll = proposal, proposed_ll
                    accepts += 1
            if iteration >= warmup:
                draws.append(theta)
        results.append({
            "chain": chain,
            "draws": draws,
            "acceptance_rate": accepts / total_iterations,
        })
    return results
