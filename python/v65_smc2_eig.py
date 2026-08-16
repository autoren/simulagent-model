#!/usr/bin/env python3
"""Sequential SMC² and pooled particle EIG for the frozen V65 external family."""
from __future__ import annotations

import copy
import hashlib
import json
import math
import random
import time
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from v22r2_grounding import PROJECT_ROOT
from v64_external_eig import V64Family, action_index, observation_index, true_transition


def load_config(path: str | Path = "configs/v65r1-design-lock.json") -> dict[str, Any]:
    value = Path(path)
    if not value.is_absolute():
        value = PROJECT_ROOT / value
    return json.loads(value.read_text())["config_payload"]


def stable_seed(base_seed: int, *parts: Any) -> int:
    payload = json.dumps([base_seed, *parts], sort_keys=True, separators=(",", ":"))
    return int.from_bytes(hashlib.sha256(payload.encode()).digest()[:8], "big")


def stream_id(base_seed: int, *parts: Any) -> str:
    payload = json.dumps([base_seed, *parts], sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def logsumexp(values: Iterable[float]) -> float:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    if not finite:
        return -math.inf
    maximum = max(finite)
    return maximum + math.log(sum(math.exp(value - maximum) for value in finite))


def normalize_log_weights(values: Sequence[float]) -> np.ndarray:
    normalizer = logsumexp(values)
    if not math.isfinite(normalizer):
        raise RuntimeError("all V65 particle weights became extinct")
    result = np.asarray(
        [0.0 if not math.isfinite(value) else math.exp(value - normalizer) for value in values],
        dtype=np.float64,
    )
    result /= result.sum()
    return result


def scaled_beta_sample(seed: int, low: float, high: float) -> float:
    return low + (high - low) * random.Random(seed).betavariate(2.0, 2.0)


def scaled_beta_log_pdf(theta: float, low: float, high: float) -> float:
    if not low < theta < high:
        return -math.inf
    unit = (theta - low) / (high - low)
    return math.log(6.0) + math.log(unit) + math.log1p(-unit) - math.log(high - low)


def theta_to_logit(theta: float, low: float, high: float) -> float:
    unit = (theta - low) / (high - low)
    return math.log(unit) - math.log1p(-unit)


def logit_to_theta(value: float, low: float, high: float) -> float:
    if value >= 0.0:
        unit = 1.0 / (1.0 + math.exp(-value))
    else:
        exp_value = math.exp(value)
        unit = exp_value / (1.0 + exp_value)
    return low + (high - low) * unit


def log_abs_theta_jacobian(theta: float, low: float, high: float) -> float:
    unit = (theta - low) / (high - low)
    return math.log(high - low) + math.log(unit) + math.log1p(-unit)


def sample_categorical_rows(probabilities: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    values = np.asarray(probabilities, dtype=np.float64)
    if values.ndim == 1:
        values = values[None, :]
    if np.min(values) < -1e-15 or np.max(np.abs(values.sum(axis=1) - 1.0)) > 1e-10:
        raise ValueError("invalid V65 categorical probabilities")
    uniforms = rng.random(values.shape[0])
    return np.minimum(
        values.shape[1] - 1,
        np.sum(uniforms[:, None] > np.cumsum(values, axis=1), axis=1),
    ).astype(np.int16)


def systematic_indices(
    weights: Sequence[float], count: int, seed: int
) -> tuple[np.ndarray, float]:
    values = np.asarray(weights, dtype=np.float64)
    values /= values.sum()
    offset = random.Random(seed).random() / count
    positions = offset + np.arange(count, dtype=np.float64) / count
    indices = np.searchsorted(np.cumsum(values), positions, side="right")
    return np.minimum(indices, len(values) - 1).astype(np.int64), offset


class StreamTracker:
    def __init__(self) -> None:
        self.total = 0
        self.collisions = 0
        self._seen: set[str] = set()

    def register(self, identifier: str) -> None:
        self.total += 1
        if identifier in self._seen:
            self.collisions += 1
        else:
            self._seen.add(identifier)

    @property
    def unique(self) -> int:
        return len(self._seen)


def _new_work() -> dict[str, int]:
    return {
        "outer_particles_initialized": 0,
        "inner_initial_draw_count": 0,
        "inner_transition_draw_count": 0,
        "observation_weight_evaluation_count": 0,
        "complete_history_likelihood_recomputation_count": 0,
        "inner_resampling_count": 0,
        "outer_resampling_count": 0,
        "pmmh_attempt_count": 0,
        "pmmh_accept_count": 0,
    }


def _merge_work(target: dict[str, int], source: dict[str, int]) -> None:
    for key, value in source.items():
        target[key] = target.get(key, 0) + int(value)


def _assert_public_record(family: V64Family, record: dict[str, Any]) -> None:
    allowed = {"record_id", "prefix_length", "initial_observation", "actions", "observations"}
    unknown = set(record) - allowed
    if unknown:
        raise PermissionError(f"V65 public record contains undeclared fields: {sorted(unknown)}")
    if set(record) != allowed:
        raise ValueError("V65 public record omits a required field")
    if int(record["prefix_length"]) != len(record["actions"]):
        raise ValueError("V65 prefix length does not match the public action history")
    if len(record["actions"]) != len(record["observations"]):
        raise ValueError("V65 history has unequal action and observation lengths")
    observation_index(family, record["initial_observation"])
    for action, observation in zip(record["actions"], record["observations"], strict=True):
        action_index(family, action)
        observation_index(family, observation)


def attempted_outcome_leak(record: dict[str, Any], realized_outcome: object) -> None:
    if any(key in record for key in ("identity", "theta", "state", "truth", "audit")):
        raise PermissionError("V65 candidate cannot read truth fields")
    if realized_outcome is not None:
        raise PermissionError("V65 selection cannot read a post-selection outcome")


def _stream_parts(
    record_id: str,
    identity: int,
    outer_budget: int,
    repeat: int,
    slot: int,
    phase: str,
    tick: int,
    purpose: str,
    extra: Sequence[Any],
    shared_inner_stream: bool,
) -> tuple[Any, ...]:
    outer_slot: Any = "shared" if shared_inner_stream and purpose.startswith("inner") else slot
    return (
        "v65",
        record_id,
        identity,
        outer_budget,
        repeat,
        outer_slot,
        phase,
        tick,
        purpose,
        *extra,
    )


def _maybe_resample_inner(
    states: np.ndarray,
    weights: np.ndarray,
    config: dict[str, Any],
    stream_parts: tuple[Any, ...],
    tracker: StreamTracker,
    work: dict[str, int],
) -> tuple[np.ndarray, np.ndarray]:
    count = len(states)
    ess = 1.0 / float(np.sum(np.square(weights)))
    if ess >= float(config["smcSquared"]["innerEssThresholdFraction"]) * count:
        return states, weights
    base = int(config["seeds"]["resamplingSeed"])
    identifier = stream_id(base, *stream_parts, "inner-systematic")
    tracker.register(identifier)
    indices, _ = systematic_indices(weights, count, stable_seed(base, *stream_parts))
    work["inner_resampling_count"] += 1
    return states[indices].copy(), np.full(count, 1.0 / count, dtype=np.float64)


def _initial_filter(
    family: V64Family,
    identity: int,
    theta: float,
    initial_observation: int,
    config: dict[str, Any],
    parts: tuple[Any, ...],
    tracker: StreamTracker,
    work: dict[str, int],
    *,
    likelihood_power: float,
    omit_reset_observation: bool,
) -> tuple[float, np.ndarray, np.ndarray]:
    del identity, theta
    count = int(config["smcSquared"]["innerStateParticleBudget"])
    base = int(config["seeds"]["innerParticleSeed"])
    identifier = stream_id(base, *parts, "inner-initial-draw")
    tracker.register(identifier)
    rng = np.random.default_rng(stable_seed(base, *parts, "initial"))
    states = sample_categorical_rows(np.repeat(family.model.initial[None, :], count, axis=0), rng)
    work["inner_initial_draw_count"] += count
    if omit_reset_observation:
        return 0.0, states, np.full(count, 1.0 / count, dtype=np.float64)
    likelihood = np.power(
        family.model.observation[0, states, initial_observation], likelihood_power
    )
    work["observation_weight_evaluation_count"] += count
    increment = float(np.mean(likelihood))
    if increment <= 0.0 or not math.isfinite(increment):
        return -math.inf, states, np.zeros(count, dtype=np.float64)
    weights = likelihood / float(likelihood.sum())
    states, weights = _maybe_resample_inner(
        states, weights, config, parts, tracker, work
    )
    return math.log(increment), states, weights


def _transition_matrix(
    family: V64Family,
    identity: int,
    theta: float,
    action: int,
    *,
    wrong_permutation: bool,
) -> np.ndarray:
    used_identity = 1 - identity if wrong_permutation else identity
    return true_transition(family, used_identity, theta, action)


def _advance_filter(
    family: V64Family,
    identity: int,
    theta: float,
    states: np.ndarray,
    weights: np.ndarray,
    action: int,
    observation: int,
    config: dict[str, Any],
    parts: tuple[Any, ...],
    tracker: StreamTracker,
    work: dict[str, int],
    *,
    likelihood_power: float,
    wrong_permutation: bool,
    observation_action_override: int | None,
) -> tuple[float, np.ndarray, np.ndarray]:
    transition = _transition_matrix(
        family, identity, theta, action, wrong_permutation=wrong_permutation
    )
    base = int(config["seeds"]["innerParticleSeed"])
    identifier = stream_id(base, *parts, "inner-transition-draw")
    tracker.register(identifier)
    rng = np.random.default_rng(stable_seed(base, *parts, "transition"))
    successors = sample_categorical_rows(transition[np.asarray(states, dtype=int)], rng)
    work["inner_transition_draw_count"] += len(states)
    observation_action = action if observation_action_override is None else observation_action_override
    likelihood = np.power(
        family.model.observation[observation_action, successors, observation],
        likelihood_power,
    )
    work["observation_weight_evaluation_count"] += len(states)
    raw = weights * likelihood
    increment = float(raw.sum())
    if increment <= 0.0 or not math.isfinite(increment):
        return -math.inf, successors, np.zeros_like(weights)
    updated = raw / increment
    successors, updated = _maybe_resample_inner(
        successors, updated, config, parts, tracker, work
    )
    return math.log(increment), successors, updated


def _full_filter(
    family: V64Family,
    identity: int,
    theta: float,
    record: dict[str, Any],
    prefix_actions: int,
    config: dict[str, Any],
    key: tuple[Any, ...],
    tracker: StreamTracker,
    work: dict[str, int],
    *,
    likelihood_power: float,
    omit_reset_observation: bool,
    wrong_permutation: bool,
    observation_action_override: int | None,
    shared_inner_stream: bool,
    count_recomputation: bool,
) -> tuple[float, np.ndarray, np.ndarray]:
    if count_recomputation:
        work["complete_history_likelihood_recomputation_count"] += 1
    record_id = str(record["record_id"])
    identity_key, outer_budget, repeat, slot, phase, *extra = key
    if int(identity_key) != identity:
        raise RuntimeError("V65 full-filter stream identity mismatch")
    parts = _stream_parts(
        record_id,
        identity,
        int(outer_budget),
        int(repeat),
        int(slot),
        str(phase),
        -1,
        "inner-reset",
        extra,
        shared_inner_stream,
    )
    log_likelihood, states, weights = _initial_filter(
        family,
        identity,
        theta,
        observation_index(family, record["initial_observation"]),
        config,
        parts,
        tracker,
        work,
        likelihood_power=likelihood_power,
        omit_reset_observation=omit_reset_observation,
    )
    if not math.isfinite(log_likelihood):
        return log_likelihood, states, weights
    for tick in range(prefix_actions):
        action = action_index(family, record["actions"][tick])
        observation = observation_index(family, record["observations"][tick])
        parts = _stream_parts(
            record_id,
            identity,
            int(outer_budget),
            int(repeat),
            int(slot),
            str(phase),
            tick,
            "inner-update",
            extra,
            shared_inner_stream,
        )
        increment, states, weights = _advance_filter(
            family,
            identity,
            theta,
            states,
            weights,
            action,
            observation,
            config,
            parts,
            tracker,
            work,
            likelihood_power=likelihood_power,
            wrong_permutation=wrong_permutation,
            observation_action_override=observation_action_override,
        )
        if not math.isfinite(increment):
            return -math.inf, states, weights
        log_likelihood += increment
    return log_likelihood, states, weights


def _copy_particle(particle: dict[str, Any]) -> dict[str, Any]:
    result = dict(particle)
    result["states"] = np.asarray(particle["states"], dtype=np.int16).copy()
    result["state_weights"] = np.asarray(
        particle["state_weights"], dtype=np.float64
    ).copy()
    return result


def _outer_identity_smc(
    family: V64Family,
    identity: int,
    record: dict[str, Any],
    config: dict[str, Any],
    outer_budget: int,
    repeat: int,
    tracker: StreamTracker,
    *,
    disable_outer_resampling: bool,
    disable_rejuvenation: bool,
    shared_inner_stream: bool,
    likelihood_power: float,
    omit_reset_observation: bool,
    wrong_permutation: bool,
    observation_action_override: int | None,
) -> dict[str, Any]:
    work = _new_work()
    low, high = map(float, config["externalFamily"]["thetaSupport"])
    record_id = str(record["record_id"])
    particles: list[dict[str, Any]] = []
    for slot in range(outer_budget):
        seed = stable_seed(
            int(config["seeds"]["outerParticleSeed"]),
            "v65",
            record_id,
            identity,
            outer_budget,
            repeat,
            slot,
            "theta-prior",
        )
        particles.append(
            {
                "theta": scaled_beta_sample(seed, low, high),
                "weight": 1.0 / outer_budget,
                "log_likelihood": 0.0,
                "states": np.asarray([], dtype=np.int16),
                "state_weights": np.asarray([], dtype=np.float64),
                "ancestor": slot,
            }
        )
    work["outer_particles_initialized"] += outer_budget
    log_evidence = 0.0
    resampling_ordinal = 0
    outer_ess_fractions: list[float] = []

    for tick in range(-1, len(record["actions"])):
        log_unnormalized: list[float] = []
        for slot, particle in enumerate(particles):
            if tick == -1:
                parts = _stream_parts(
                    record_id,
                    identity,
                    outer_budget,
                    repeat,
                    slot,
                    "update",
                    -1,
                    "inner-reset",
                    (),
                    shared_inner_stream,
                )
                increment, states, weights = _initial_filter(
                    family,
                    identity,
                    float(particle["theta"]),
                    observation_index(family, record["initial_observation"]),
                    config,
                    parts,
                    tracker,
                    work,
                    likelihood_power=likelihood_power,
                    omit_reset_observation=omit_reset_observation,
                )
            else:
                action = action_index(family, record["actions"][tick])
                observation = observation_index(family, record["observations"][tick])
                parts = _stream_parts(
                    record_id,
                    identity,
                    outer_budget,
                    repeat,
                    slot,
                    "update",
                    tick,
                    "inner-update",
                    (),
                    shared_inner_stream,
                )
                increment, states, weights = _advance_filter(
                    family,
                    identity,
                    float(particle["theta"]),
                    np.asarray(particle["states"]),
                    np.asarray(particle["state_weights"]),
                    action,
                    observation,
                    config,
                    parts,
                    tracker,
                    work,
                    likelihood_power=likelihood_power,
                    wrong_permutation=wrong_permutation,
                    observation_action_override=observation_action_override,
                )
            if not math.isfinite(increment):
                log_unnormalized.append(-math.inf)
                continue
            particle["states"] = states
            particle["state_weights"] = weights
            particle["log_likelihood"] = float(particle["log_likelihood"]) + increment
            previous_weight = float(particle["weight"])
            log_unnormalized.append(
                -math.inf if previous_weight <= 0.0 else math.log(previous_weight) + increment
            )

        evidence_increment = logsumexp(log_unnormalized)
        if not math.isfinite(evidence_increment):
            raise RuntimeError("all V65 outer particles became extinct")
        log_evidence += evidence_increment
        normalized = normalize_log_weights(log_unnormalized)
        for particle, weight in zip(particles, normalized, strict=True):
            particle["weight"] = float(weight)
        ess = 1.0 / float(np.sum(np.square(normalized)))
        outer_ess_fractions.append(ess / outer_budget)

        if (
            disable_outer_resampling
            or ess >= float(config["smcSquared"]["outerEssThresholdFraction"]) * outer_budget
        ):
            continue
        base = int(config["seeds"]["resamplingSeed"])
        resample_parts = (
            "v65",
            record_id,
            identity,
            outer_budget,
            repeat,
            tick,
            "outer-systematic",
            resampling_ordinal,
        )
        tracker.register(stream_id(base, *resample_parts))
        indices, _ = systematic_indices(
            normalized, outer_budget, stable_seed(base, *resample_parts)
        )
        particles = [_copy_particle(particles[int(index)]) for index in indices]
        for particle in particles:
            particle["weight"] = 1.0 / outer_budget
        work["outer_resampling_count"] += 1

        if not disable_rejuvenation:
            for slot, particle in enumerate(particles):
                for move in range(
                    int(config["smcSquared"]["rejuvenationStepsPerOuterResampling"])
                ):
                    work["pmmh_attempt_count"] += 1
                    proposal_seed = stable_seed(
                        int(config["seeds"]["rejuvenationSeed"]),
                        "v65",
                        record_id,
                        identity,
                        outer_budget,
                        repeat,
                        slot,
                        tick,
                        resampling_ordinal,
                        move,
                        "proposal",
                    )
                    proposal_rng = random.Random(proposal_seed)
                    current_theta = float(particle["theta"])
                    proposal_theta = logit_to_theta(
                        theta_to_logit(current_theta, low, high)
                        + proposal_rng.gauss(
                            0.0, float(config["smcSquared"]["proposalStandardDeviation"])
                        ),
                        low,
                        high,
                    )
                    proposed_ll, states, state_weights = _full_filter(
                        family,
                        identity,
                        proposal_theta,
                        record,
                        tick + 1,
                        config,
                        (
                            identity,
                            outer_budget,
                            repeat,
                            slot,
                            "move",
                            tick,
                            resampling_ordinal,
                            move,
                        ),
                        tracker,
                        work,
                        likelihood_power=likelihood_power,
                        omit_reset_observation=omit_reset_observation,
                        wrong_permutation=wrong_permutation,
                        observation_action_override=observation_action_override,
                        shared_inner_stream=shared_inner_stream,
                        count_recomputation=True,
                    )
                    if not math.isfinite(proposed_ll):
                        continue
                    current_target = (
                        scaled_beta_log_pdf(current_theta, low, high)
                        + float(particle["log_likelihood"])
                        + log_abs_theta_jacobian(current_theta, low, high)
                    )
                    proposed_target = (
                        scaled_beta_log_pdf(proposal_theta, low, high)
                        + proposed_ll
                        + log_abs_theta_jacobian(proposal_theta, low, high)
                    )
                    accept_seed = stable_seed(
                        int(config["seeds"]["rejuvenationSeed"]),
                        "v65",
                        record_id,
                        identity,
                        outer_budget,
                        repeat,
                        slot,
                        tick,
                        resampling_ordinal,
                        move,
                        "accept",
                    )
                    log_uniform = math.log(
                        max(random.Random(accept_seed).random(), 1e-300)
                    )
                    if log_uniform < min(0.0, proposed_target - current_target):
                        particle.update(
                            {
                                "theta": proposal_theta,
                                "log_likelihood": proposed_ll,
                                "states": states,
                                "state_weights": state_weights,
                            }
                        )
                        work["pmmh_accept_count"] += 1
        resampling_ordinal += 1

    return {
        "identity": identity,
        "log_evidence": log_evidence,
        "particles": particles,
        "outer_ess_fractions": outer_ess_fractions,
        "work": work,
    }


def _state_distribution(states: np.ndarray, weights: np.ndarray, state_count: int) -> np.ndarray:
    result = np.zeros(state_count, dtype=np.float64)
    np.add.at(result, np.asarray(states, dtype=int), np.asarray(weights, dtype=np.float64))
    total = float(result.sum())
    if total <= 0.0:
        raise RuntimeError("V65 inner posterior has no mass")
    return result / total


def canonicalize_atoms(atoms: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[int, str], dict[str, Any]] = {}
    for atom in atoms:
        identity = int(atom["identity"])
        theta = float(atom["theta"])
        weight = float(atom["weight"])
        state = np.asarray(atom["state"], dtype=np.float64)
        if weight <= 0.0:
            continue
        if not math.isfinite(weight) or np.any(~np.isfinite(state)):
            raise ValueError("V65 particle measure contains a non-finite atom")
        key = (identity, theta.hex())
        if key not in grouped:
            grouped[key] = {
                "identity": identity,
                "theta": theta,
                "weight": 0.0,
                "state_numerator": np.zeros_like(state),
            }
        grouped[key]["weight"] += weight
        grouped[key]["state_numerator"] += weight * state
    result: list[dict[str, Any]] = []
    total = sum(float(row["weight"]) for row in grouped.values())
    if total <= 0.0:
        raise RuntimeError("V65 particle measure has no mass")
    for row in grouped.values():
        mass = float(row["weight"])
        state = np.asarray(row.pop("state_numerator")) / mass
        result.append(
            {
                "identity": int(row["identity"]),
                "theta": float(row["theta"]),
                "weight": mass / total,
                "state": state / state.sum(),
            }
        )
    result.sort(key=lambda row: (row["identity"], row["theta"]))
    return result


def smc2_inference(
    family: V64Family,
    record: dict[str, Any],
    config: dict[str, Any],
    outer_budget: int,
    repeat: int,
    *,
    disable_outer_resampling: bool = False,
    disable_rejuvenation: bool = False,
    shared_inner_stream: bool = False,
    likelihood_power: float = 1.0,
    omit_reset_observation: bool = False,
    wrong_permutation: bool = False,
    observation_action_override: int | None = None,
    equal_identity_evidence: bool = False,
) -> dict[str, Any]:
    started = time.perf_counter()
    _assert_public_record(family, record)
    if outer_budget <= 0 or repeat < 0:
        raise ValueError("invalid V65 outer budget or repeat")
    tracker = StreamTracker()
    identity_results = [
        _outer_identity_smc(
            family,
            identity,
            record,
            config,
            outer_budget,
            repeat,
            tracker,
            disable_outer_resampling=disable_outer_resampling,
            disable_rejuvenation=disable_rejuvenation,
            shared_inner_stream=shared_inner_stream,
            likelihood_power=likelihood_power,
            omit_reset_observation=omit_reset_observation,
            wrong_permutation=wrong_permutation,
            observation_action_override=observation_action_override,
        )
        for identity in range(2)
    ]
    if equal_identity_evidence:
        identity_mass = np.asarray([0.5, 0.5], dtype=np.float64)
    else:
        identity_mass = normalize_log_weights(
            [row["log_evidence"] - math.log(2.0) for row in identity_results]
        )
    atoms: list[dict[str, Any]] = []
    work = _new_work()
    ess_values: list[float] = []
    for identity, (result, identity_weight) in enumerate(
        zip(identity_results, identity_mass, strict=True)
    ):
        _merge_work(work, result["work"])
        ess_values.extend(result["outer_ess_fractions"])
        for particle in result["particles"]:
            atoms.append(
                {
                    "identity": identity,
                    "theta": float(particle["theta"]),
                    "weight": float(identity_weight) * float(particle["weight"]),
                    "state": _state_distribution(
                        np.asarray(particle["states"]),
                        np.asarray(particle["state_weights"]),
                        len(family.model.states),
                    ),
                }
            )
    atoms = canonicalize_atoms(atoms)
    runtime = time.perf_counter() - started
    work["final_posterior_atom_count"] = len(atoms)
    return {
        "record_id": record["record_id"],
        "outer_budget": int(outer_budget),
        "repeat": int(repeat),
        "atoms": atoms,
        "identity": posterior_summary(family, atoms)["identity"],
        "log_evidence_by_identity": [
            float(row["log_evidence"]) for row in identity_results
        ],
        "normalizes": abs(sum(float(atom["weight"]) for atom in atoms) - 1.0) <= 1e-10,
        "diagnostics": {
            "work": work,
            "runtime_seconds": runtime,
            "random_stream_count": tracker.total,
            "unique_random_stream_count": tracker.unique,
            "random_stream_collision_count": tracker.collisions,
            "mean_outer_ess_fraction": float(np.mean(ess_values)) if ess_values else 1.0,
            "minimum_outer_ess_fraction": float(np.min(ess_values)) if ess_values else 1.0,
        },
    }


def pool_repeats(repeats: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if not repeats:
        raise ValueError("V65 repeat pool cannot be empty")
    expected_budget = int(repeats[0]["outer_budget"])
    expected_record = str(repeats[0]["record_id"])
    if any(
        int(row["outer_budget"]) != expected_budget
        or str(row["record_id"]) != expected_record
        or not row["normalizes"]
        for row in repeats
    ):
        raise ValueError("V65 repeat pool mixes records, budgets, or unnormalized measures")
    scale = 1.0 / len(repeats)
    atoms = canonicalize_atoms(
        [
            {
                "identity": atom["identity"],
                "theta": atom["theta"],
                "state": np.asarray(atom["state"]),
                "weight": scale * float(atom["weight"]),
            }
            for repeat in repeats
            for atom in repeat["atoms"]
        ]
    )
    return {
        "record_id": expected_record,
        "outer_budget": expected_budget,
        "repeat_count": len(repeats),
        "pooling_rule": "equal_weight_posterior_mixture_before_scoring",
        "atoms": atoms,
        "normalizes": abs(sum(float(atom["weight"]) for atom in atoms) - 1.0) <= 1e-10,
    }


def _exact_conditional_state(
    family: V64Family,
    identity: int,
    theta: float,
    record: dict[str, Any],
) -> np.ndarray:
    state = np.asarray(family.model.initial, dtype=np.float64).copy()
    initial_observation = observation_index(family, record["initial_observation"])
    state *= family.model.observation[0, :, initial_observation]
    normalizer = float(state.sum())
    if normalizer <= 0.0:
        raise ValueError("impossible V65r1 reset observation")
    state /= normalizer
    for action_name, observation_name in zip(
        record["actions"], record["observations"], strict=True
    ):
        action = action_index(family, action_name)
        observation = observation_index(family, observation_name)
        state = state @ true_transition(family, identity, theta, action)
        state *= family.model.observation[action, :, observation]
        normalizer = float(state.sum())
        if normalizer <= 0.0:
            raise ValueError("impossible V65r1 point-parameter public history")
        state /= normalizer
    return state


def rao_blackwellize_measure(
    family: V64Family,
    measure: dict[str, Any],
    record: dict[str, Any],
    *,
    allow_unpooled_fixture: bool = False,
) -> dict[str, Any]:
    """Replace only the dynamic state conditional, preserving all static SMC² weights."""
    _assert_public_record(family, record)
    if not allow_unpooled_fixture and (
        int(measure.get("repeat_count", 0)) != 3
        or measure.get("pooling_rule") != "equal_weight_posterior_mixture_before_scoring"
    ):
        raise ValueError("V65r1 acquisition must pool three repeats before Rao-Blackwellization")
    atoms = canonicalize_atoms(
        [
            {
                **atom,
                "state": _exact_conditional_state(
                    family,
                    int(atom["identity"]),
                    float(atom["theta"]),
                    record,
                ),
            }
            for atom in measure["atoms"]
        ]
    )
    return {
        **measure,
        "atoms": atoms,
        "rao_blackwellized_known_state": True,
        "static_weights_unchanged": True,
        "particle_state_predictive_role": "negative_control_and_diagnostic_only",
    }


def _atom_observation_predictive(
    family: V64Family,
    atom: dict[str, Any],
    action: int,
    *,
    wrong_permutation: bool = False,
    observation_action_override: int | None = None,
) -> np.ndarray:
    identity = int(atom["identity"])
    theta = float(atom["theta"])
    transition = _transition_matrix(
        family, identity, theta, action, wrong_permutation=wrong_permutation
    )
    observation_action = action if observation_action_override is None else observation_action_override
    state = np.asarray(atom["state"], dtype=np.float64)
    result = (state @ transition) @ family.model.observation[observation_action]
    result = np.asarray(result, dtype=np.float64)
    result /= result.sum()
    return result


def score_action(
    family: V64Family,
    measure: dict[str, Any] | Sequence[dict[str, Any]],
    action: int | str,
    *,
    wrong_permutation: bool = False,
    observation_action_override: int | None = None,
) -> dict[str, Any]:
    action_id = action_index(family, action)
    source_atoms = measure["atoms"] if isinstance(measure, dict) else measure
    atoms = canonicalize_atoms(source_atoms)
    weights = np.asarray([float(atom["weight"]) for atom in atoms], dtype=np.float64)
    conditional = np.asarray(
        [
            _atom_observation_predictive(
                family,
                atom,
                action_id,
                wrong_permutation=wrong_permutation,
                observation_action_override=observation_action_override,
            )
            for atom in atoms
        ]
    )
    predictive = weights @ conditional
    information = 0.0
    for atom_weight, atom_predictive in zip(weights, conditional, strict=True):
        mask = atom_predictive > 0.0
        information += float(
            atom_weight
            * np.sum(
                atom_predictive[mask]
                * np.log(atom_predictive[mask] / predictive[mask])
            )
        )
    return {
        "action": family.model.actions[action_id],
        "action_index": action_id,
        "eig": float(information),
        "predictive": predictive.tolist(),
        "normalizes": abs(float(predictive.sum()) - 1.0) <= 1e-12,
        "finite": bool(math.isfinite(information) and np.all(np.isfinite(predictive))),
        "static_atom_count": len(atoms),
    }


def score_all_actions(
    family: V64Family, measure: dict[str, Any] | Sequence[dict[str, Any]]
) -> list[dict[str, Any]]:
    scores = [score_action(family, measure, action) for action in family.canonical_actions]
    if [row["action"] for row in scores] != ["n", "e", "s", "w"]:
        raise RuntimeError("V65 candidate omission or canonical-order violation")
    return scores


def select_action(
    family: V64Family,
    measure: dict[str, Any] | Sequence[dict[str, Any]],
    *,
    tolerance: float = 1e-12,
) -> dict[str, Any]:
    scores = score_all_actions(family, measure)
    maximum = max(float(row["eig"]) for row in scores)
    optimal = [row for row in scores if float(row["eig"]) >= maximum - tolerance]
    if not optimal:
        raise RuntimeError("V65 approximate selection has no finite optimum")
    return {
        "selected": optimal[0],
        "maximum": maximum,
        "optimal_actions": [row["action"] for row in optimal],
        "scores": scores,
    }


def posterior_summary(
    family: V64Family, measure: dict[str, Any] | Sequence[dict[str, Any]], bins: int = 16
) -> dict[str, Any]:
    source_atoms = measure["atoms"] if isinstance(measure, dict) else measure
    atoms = canonicalize_atoms(source_atoms)
    identity = np.zeros(2, dtype=np.float64)
    state = np.zeros(len(family.model.states), dtype=np.float64)
    theta_values: list[float] = []
    theta_weights: list[float] = []
    joint_bins = np.zeros((2, bins), dtype=np.float64)
    low, high = family.theta_support
    for atom in atoms:
        identity_id = int(atom["identity"])
        theta = float(atom["theta"])
        weight = float(atom["weight"])
        identity[identity_id] += weight
        state += weight * np.asarray(atom["state"], dtype=np.float64)
        theta_values.append(theta)
        theta_weights.append(weight)
        theta_bin = min(bins - 1, max(0, int((theta - low) / (high - low) * bins)))
        joint_bins[identity_id, theta_bin] += weight
    return {
        "identity": identity,
        "theta_values": np.asarray(theta_values, dtype=np.float64),
        "theta_weights": np.asarray(theta_weights, dtype=np.float64),
        "joint_bins": joint_bins,
        "state": state,
        "normalizes": bool(
            abs(float(identity.sum()) - 1.0) <= 1e-12
            and abs(float(state.sum()) - 1.0) <= 1e-12
            and abs(float(joint_bins.sum()) - 1.0) <= 1e-12
        ),
    }


def score_state_as_target(
    family: V64Family,
    measure: dict[str, Any] | Sequence[dict[str, Any]],
    action: int | str,
) -> float:
    action_id = action_index(family, action)
    source_atoms = measure["atoms"] if isinstance(measure, dict) else measure
    target_rows: list[tuple[float, np.ndarray]] = []
    for atom in canonicalize_atoms(source_atoms):
        transition = _transition_matrix(
            family,
            int(atom["identity"]),
            float(atom["theta"]),
            action_id,
            wrong_permutation=False,
        )
        for state_id, state_mass in enumerate(np.asarray(atom["state"])):
            weight = float(atom["weight"]) * float(state_mass)
            if weight <= 0.0:
                continue
            conditional = transition[state_id] @ family.model.observation[action_id]
            target_rows.append((weight, conditional))
    predictive = sum(weight * conditional for weight, conditional in target_rows)
    information = 0.0
    for weight, conditional in target_rows:
        mask = conditional > 0.0
        information += weight * float(
            np.sum(conditional[mask] * np.log(conditional[mask] / predictive[mask]))
        )
    return float(information)


def collapse_map_identity(measure: dict[str, Any]) -> dict[str, Any]:
    atoms = canonicalize_atoms(measure["atoms"])
    masses = np.zeros(2, dtype=np.float64)
    for atom in atoms:
        masses[int(atom["identity"])] += float(atom["weight"])
    selected = int(np.argmax(masses))
    kept = [copy.deepcopy(atom) for atom in atoms if int(atom["identity"]) == selected]
    for atom in kept:
        atom["weight"] = float(atom["weight"]) / float(masses[selected])
    return {**measure, "atoms": canonicalize_atoms(kept), "control": "map_identity"}


def collapse_theta_mean(measure: dict[str, Any]) -> dict[str, Any]:
    atoms = canonicalize_atoms(measure["atoms"])
    mean = sum(float(atom["weight"]) * float(atom["theta"]) for atom in atoms)
    collapsed = [
        {
            "identity": atom["identity"],
            "theta": mean,
            "weight": atom["weight"],
            "state": np.asarray(atom["state"]),
        }
        for atom in atoms
    ]
    return {**measure, "atoms": canonicalize_atoms(collapsed), "control": "theta_mean"}


def force_equal_identity_evidence(measure: dict[str, Any]) -> dict[str, Any]:
    atoms = canonicalize_atoms(measure["atoms"])
    masses = np.zeros(2, dtype=np.float64)
    for atom in atoms:
        masses[int(atom["identity"])] += float(atom["weight"])
    adjusted = []
    for atom in atoms:
        identity = int(atom["identity"])
        if masses[identity] <= 0.0:
            continue
        adjusted.append(
            {
                **atom,
                "weight": 0.5 * float(atom["weight"]) / float(masses[identity]),
            }
        )
    return {
        **measure,
        "atoms": canonicalize_atoms(adjusted),
        "control": "equal_identity_evidence",
    }
