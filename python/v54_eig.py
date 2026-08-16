"""Exact one-step expected-information-gain utilities for V54."""
from __future__ import annotations

import json
import math
from collections import defaultdict
from fractions import Fraction
from typing import Any, Iterable, Sequence

from v22_relational import canonical_json
from v42_stateful import action_bindings, atom_universe
from v46_stochastic import _configuration_key
from v49_belief import _configuration_key_with_history, masked_trace, trace_key
from v53_smc2 import continuous_advance_configurations, instantiate_program


def candidate_interventions(entity_rows: Sequence[dict[str, str]]):
    """Enumerate every legal bound action and the wait action, canonically."""
    actions = [{"id": "wait", "binding": {}}]
    for action_id in ("pulse", "route"):
        actions.extend({"id": action_id, "binding": dict(binding)} for binding in action_bindings(entity_rows))
    interventions = []
    for action in actions:
        assay = [
            action,
            {"id": "wait", "binding": {}},
            {"id": "wait", "binding": {}},
        ]
        interventions.append({
            "action": action,
            "assay": assay,
            "key": canonical_json(assay),
        })
    interventions.sort(key=lambda row: row["key"])
    expected = 1 + 2 * len(entity_rows) * (len(entity_rows) - 1)
    if len(interventions) != expected or len({row["key"] for row in interventions}) != expected:
        raise RuntimeError("V54 candidate intervention set is incomplete or duplicated")
    return interventions


def _decode_configuration(key: str):
    value = json.loads(key)
    return dict(value["world"]), list(value["queue"])


def belief_atoms_from_exact(exact):
    """Retain the joint quadrature target and hidden configurations."""
    atoms = []
    for row, target_weight in zip(exact["rows"], exact["weights"], strict=True):
        if not target_weight or row["result"] is None:
            continue
        for configuration_key, conditional_weight in row["result"]["configuration"].items():
            weight = target_weight * conditional_weight
            if not weight:
                continue
            world, queue = _decode_configuration(configuration_key)
            atoms.append({
                "program_index": row["program_index"],
                "node_index": row["node_index"],
                "theta": row["theta"],
                "configuration_key": configuration_key,
                "world": world,
                "queue": queue,
                "weight": weight,
            })
    total = sum(row["weight"] for row in atoms)
    if total <= 0:
        raise RuntimeError("V54 exact belief has zero mass")
    for row in atoms:
        row["weight"] /= total
    if abs(sum(row["weight"] for row in atoms) - 1.0) > 1e-12:
        raise RuntimeError("V54 exact belief atoms do not normalize")
    return atoms


def prior_belief_atoms(registry, quadrature, world, queue=None):
    """Build an exact factorized prior fixture at a known configuration."""
    queue = [] if queue is None else list(queue)
    configuration_key = _configuration_key(world, queue)
    atoms = []
    for program_index in range(len(registry)):
        for node_index, (theta, theta_weight) in enumerate(quadrature):
            atoms.append({
                "program_index": program_index,
                "node_index": node_index,
                "theta": theta,
                "configuration_key": configuration_key,
                "world": dict(world),
                "queue": list(queue),
                "weight": theta_weight / len(registry),
            })
    return atoms


def target_key(atom) -> str:
    return f"{atom['program_index']}:{atom['node_index']}"


def _masked_outcome_key(history, entity_rows):
    masks = [list(atom_universe(entity_rows)) for _ in history]
    return trace_key(masked_trace(history, masks))


def atom_outcomes(atom, registry, entity_rows, intervention, start_tick: int):
    program = instantiate_program(
        registry[atom["program_index"]]["template"], atom["theta"]
    )
    seed = {
        _configuration_key_with_history(atom["world"], atom["queue"], []): {
            "world": dict(atom["world"]),
            "queue": list(atom["queue"]),
            "history": [],
            "mass": Fraction(1),
        }
    }
    branches = continuous_advance_configurations(
        program, entity_rows, seed, intervention["assay"], start_tick
    )
    outcomes = defaultdict(float)
    for branch in branches.values():
        key = _masked_outcome_key(branch["history"], entity_rows)
        outcomes[key] += float(branch["mass"])
    total = sum(outcomes.values())
    if abs(total - 1.0) > 1e-12:
        raise RuntimeError("V54 atom-level assay outcomes do not normalize")
    return dict(outcomes)


def joint_target_outcome(
    atoms,
    registry,
    entity_rows,
    intervention,
    start_tick: int,
    target="program_theta",
):
    """Accumulate p(target, outcome), integrating out hidden configuration."""
    joint = defaultdict(lambda: defaultdict(float))
    target_prior = defaultdict(float)
    conditional_rows = []
    for atom in atoms:
        if target == "program_theta":
            key = target_key(atom)
        elif target == "configuration":
            key = atom["configuration_key"]
        else:
            raise ValueError(f"Unknown V54 information target: {target}")
        target_prior[key] += atom["weight"]
        outcomes = atom_outcomes(
            atom, registry, entity_rows, intervention, start_tick
        )
        conditional_rows.append((key, atom["weight"], outcomes))
        for outcome, probability in outcomes.items():
            joint[outcome][key] += atom["weight"] * probability
    prior_total = sum(target_prior.values())
    joint_total = sum(sum(row.values()) for row in joint.values())
    if abs(prior_total - 1.0) > 1e-12 or abs(joint_total - 1.0) > 1e-12:
        raise RuntimeError("V54 target/outcome joint does not normalize")
    return {
        "target_prior": dict(target_prior),
        "joint": {outcome: dict(values) for outcome, values in joint.items()},
        "conditional_rows": conditional_rows,
    }


def entropy(probabilities: Iterable[float]) -> float:
    return -sum(value * math.log(value) for value in probabilities if value > 0)


def expected_information_gain_from_joint(target_prior, joint):
    """One-pass mutual information plus identity diagnostics."""
    predictive = {
        outcome: sum(values.values()) for outcome, values in joint.items()
    }
    eig = 0.0
    expected_posterior_entropy = 0.0
    for outcome, target_masses in joint.items():
        outcome_mass = predictive[outcome]
        if outcome_mass <= 0:
            continue
        posterior = []
        for key, mass in target_masses.items():
            if mass <= 0:
                continue
            prior = target_prior.get(key, 0.0)
            if prior <= 0:
                raise RuntimeError("V54 posterior mass lies outside target prior support")
            eig += mass * math.log(mass / (prior * outcome_mass))
            posterior.append(mass / outcome_mass)
        expected_posterior_entropy += outcome_mass * entropy(posterior)
    prior_entropy = entropy(target_prior.values())
    entropy_eig = prior_entropy - expected_posterior_entropy
    return {
        "eig": eig,
        "entropy_eig": entropy_eig,
        "prior_entropy": prior_entropy,
        "expected_posterior_entropy": expected_posterior_entropy,
        "predictive": predictive,
        "normalizes": (
            abs(sum(target_prior.values()) - 1.0) <= 1e-12
            and abs(sum(predictive.values()) - 1.0) <= 1e-12
        ),
        "finite": all(math.isfinite(value) for value in (
            eig, entropy_eig, prior_entropy, expected_posterior_entropy
        )),
    }


def scalar_reference_eig(target_prior, joint):
    """Reference form: enumerate outcomes, form posteriors, then average KL."""
    result = 0.0
    for target_outcome in sorted(joint):
        outcome_mass = sum(joint[target_outcome].values())
        if outcome_mass <= 0:
            continue
        posterior = {
            key: joint[target_outcome].get(key, 0.0) / outcome_mass
            for key in target_prior
        }
        divergence = 0.0
        for key in sorted(target_prior):
            probability = posterior[key]
            if probability > 0:
                divergence += probability * math.log(
                    probability / target_prior[key]
                )
        result += outcome_mass * divergence
    return result


def score_intervention(
    atoms, registry, entity_rows, intervention, start_tick: int,
    target="program_theta",
):
    distribution = joint_target_outcome(
        atoms, registry, entity_rows, intervention, start_tick, target
    )
    score = expected_information_gain_from_joint(
        distribution["target_prior"], distribution["joint"]
    )
    score["reference_eig"] = scalar_reference_eig(
        distribution["target_prior"], distribution["joint"]
    )
    score["predictive_entropy"] = entropy(score["predictive"].values())
    score["intervention_key"] = intervention["key"]
    return score


def score_all_interventions(
    atoms, registry, entity_rows, start_tick: int, target="program_theta"
):
    return [
        score_intervention(
            atoms, registry, entity_rows, intervention, start_tick, target
        )
        for intervention in candidate_interventions(entity_rows)
    ]


def select_score(scores, tolerance=1e-12, field="eig"):
    if not scores:
        raise ValueError("V54 cannot select from an empty score set")
    maximum = max(row[field] for row in scores)
    optimal = sorted(
        (row for row in scores if row[field] >= maximum - tolerance),
        key=lambda row: row["intervention_key"],
    )
    return {
        "selected": optimal[0],
        "maximum": maximum,
        "optimal_keys": [row["intervention_key"] for row in optimal],
    }


def filter_and_normalize_atoms(atoms, predicate):
    selected = [dict(row) for row in atoms if predicate(row)]
    total = sum(row["weight"] for row in selected)
    if total <= 0:
        raise RuntimeError("V54 control removed all belief mass")
    for row in selected:
        row["weight"] /= total
    return selected


def map_program_atoms(atoms):
    masses = defaultdict(float)
    for atom in atoms:
        masses[atom["program_index"]] += atom["weight"]
    selected = min(masses, key=lambda key: (-masses[key], key))
    return filter_and_normalize_atoms(
        atoms, lambda atom: atom["program_index"] == selected
    )


def theta_point_mass_atoms(atoms):
    mean = sum(atom["theta"] * atom["weight"] for atom in atoms)
    result = []
    for atom in atoms:
        row = dict(atom)
        row["theta"] = mean
        row["node_index"] = 0
        result.append(row)
    return result


def likelihood_squared_eig(score_distribution):
    """Pseudo-EIG after sharpening each p(outcome | target) and renormalizing."""
    prior = score_distribution["target_prior"]
    joint = score_distribution["joint"]
    outcomes = sorted(joint)
    pseudo_joint = {outcome: {} for outcome in outcomes}
    for target, prior_mass in prior.items():
        likelihoods = [
            joint[outcome].get(target, 0.0) / prior_mass
            if prior_mass > 0 else 0.0
            for outcome in outcomes
        ]
        normalizer = sum(value * value for value in likelihoods)
        if normalizer <= 0:
            raise RuntimeError("V54 likelihood-squared control has zero conditional mass")
        for outcome, likelihood in zip(outcomes, likelihoods, strict=True):
            pseudo_joint[outcome][target] = (
                prior_mass * likelihood * likelihood / normalizer
            )
    return expected_information_gain_from_joint(prior, pseudo_joint)["eig"]


def score_control_policies(atoms, registry, entity_rows, start_tick: int):
    """Return frozen control selections without reading truth or realized outcomes."""
    interventions = candidate_interventions(entity_rows)
    primary, state_only, squared = [], [], []
    for intervention in interventions:
        distribution = joint_target_outcome(
            atoms, registry, entity_rows, intervention, start_tick
        )
        row = expected_information_gain_from_joint(
            distribution["target_prior"], distribution["joint"]
        )
        row.update({
            "intervention_key": intervention["key"],
            "predictive_entropy": entropy(row["predictive"].values()),
        })
        primary.append(row)
        state_only.append(score_intervention(
            atoms, registry, entity_rows, intervention, start_tick,
            target="configuration",
        ))
        squared.append({
            "intervention_key": intervention["key"],
            "eig": likelihood_squared_eig(distribution),
        })
    map_scores = score_all_interventions(
        map_program_atoms(atoms), registry, entity_rows, start_tick
    )
    point_scores = score_all_interventions(
        theta_point_mass_atoms(atoms), registry, entity_rows, start_tick
    )
    return {
        "primary": select_score(primary),
        "uniform_random_mean_eig": sum(row["eig"] for row in primary) / len(primary),
        "predictive_entropy": select_score(primary, field="predictive_entropy"),
        "state_only_information": select_score(state_only),
        "map_program": select_score(map_scores),
        "theta_point_mass": select_score(point_scores),
        "likelihood_squared": select_score(squared),
    }


FORBIDDEN_SELECTION_KEYS = frozenset({
    "realized_outcome",
    "target_program_index",
    "target_program_key",
    "target_program_ordinal",
    "target_theta",
    "true_configuration_key",
    "truth",
})


def assert_selection_payload_is_public(value):
    if isinstance(value, dict):
        overlap = set(value) & FORBIDDEN_SELECTION_KEYS
        if overlap:
            raise PermissionError(
                "V54 selection payload contains forbidden fields: "
                + ", ".join(sorted(overlap))
            )
        for child in value.values():
            assert_selection_payload_is_public(child)
    elif isinstance(value, list):
        for child in value:
            assert_selection_payload_is_public(child)


def attempted_outcome_leak_selection(public_history, realized_outcome):
    del public_history, realized_outcome
    raise PermissionError(
        "V54 selection firewall rejects realized-outcome access before selection"
    )
