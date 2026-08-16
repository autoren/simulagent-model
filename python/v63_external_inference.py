"""Exact and SMC-squared inference for the V63 external Tiger family."""
from __future__ import annotations

import copy
import hashlib
import json
import math
import random
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np


LISTEN, OPEN_LEFT, OPEN_RIGHT = 0, 1, 2
LEFT_START, RIGHT_START, LEFT, RIGHT, TERMINAL = range(5)
PERSISTENT, ALTERNATING = 0, 1


def load_anchor(path: str | Path) -> dict[str, Any]:
    model = json.loads(Path(path).read_text())
    required = {
        "states", "actions", "observations", "discount", "initial",
        "transition", "observation", "reward",
    }
    if not required.issubset(model):
        raise ValueError("external anchor is missing required arrays")
    if model["states"] != [
        "tiger-left-start", "tiger-right-start", "tiger-left", "tiger-right", "terminal"
    ]:
        raise ValueError("V63 requires the pinned five-state Tiger ordering")
    if model["actions"] != ["listen", "open-left", "open-right"]:
        raise ValueError("V63 requires the pinned Tiger action ordering")
    if model["observations"] != ["init", "tiger-left", "tiger-right", "terminal"]:
        raise ValueError("V63 requires the pinned Tiger observation ordering")
    return model


def stable_seed(base_seed: int, *parts: Any) -> int:
    payload = json.dumps([base_seed, *parts], sort_keys=True, separators=(",", ":"))
    return int.from_bytes(hashlib.sha256(payload.encode()).digest()[:8], "big")


def stream_id(base_seed: int, *parts: Any) -> str:
    payload = json.dumps([base_seed, *parts], sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def scaled_beta_log_pdf(
    theta: float,
    low: float = 0.65,
    high: float = 0.95,
    alpha: float = 2.0,
    beta: float = 2.0,
) -> float:
    if not low < theta < high:
        return -math.inf
    unit = (theta - low) / (high - low)
    log_beta = math.lgamma(alpha) + math.lgamma(beta) - math.lgamma(alpha + beta)
    return (
        (alpha - 1.0) * math.log(unit)
        + (beta - 1.0) * math.log1p(-unit)
        - log_beta
        - math.log(high - low)
    )


def scaled_beta_sample(
    seed: int,
    low: float = 0.65,
    high: float = 0.95,
    alpha: float = 2.0,
    beta: float = 2.0,
) -> float:
    value = random.Random(seed).betavariate(alpha, beta)
    return low + (high - low) * value


def theta_to_logit(theta: float, low: float = 0.65, high: float = 0.95) -> float:
    unit = (theta - low) / (high - low)
    return math.log(unit) - math.log1p(-unit)


def logit_to_theta(value: float, low: float = 0.65, high: float = 0.95) -> float:
    if value >= 0.0:
        unit = 1.0 / (1.0 + math.exp(-value))
    else:
        exp_value = math.exp(value)
        unit = exp_value / (1.0 + exp_value)
    return low + (high - low) * unit


def log_abs_theta_jacobian(theta: float, low: float = 0.65, high: float = 0.95) -> float:
    unit = (theta - low) / (high - low)
    return math.log(high - low) + math.log(unit) + math.log1p(-unit)


def logsumexp(values: Iterable[float]) -> float:
    finite = [value for value in values if math.isfinite(value)]
    if not finite:
        return -math.inf
    maximum = max(finite)
    return maximum + math.log(sum(math.exp(value - maximum) for value in finite))


def normalize_log_weights(values: Sequence[float]) -> list[float]:
    normalizer = logsumexp(values)
    if not math.isfinite(normalizer):
        raise RuntimeError("all V63 hypotheses have zero mass")
    return [0.0 if not math.isfinite(value) else math.exp(value - normalizer) for value in values]


def quadrature_rule(nodes: int, parameter: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    raw_nodes, raw_weights = np.polynomial.legendre.leggauss(nodes)
    low, high = map(float, parameter["support"])
    theta = low + (raw_nodes + 1.0) * (high - low) / 2.0
    weights = raw_weights * (high - low) / 2.0
    prior = np.asarray([
        math.exp(scaled_beta_log_pdf(
            float(value), low, high, float(parameter["alpha"]), float(parameter["beta"])
        ))
        for value in theta
    ])
    weights *= prior
    weights /= weights.sum()
    return theta, weights


def family_transition(anchor: dict[str, Any], identity: int, theta: float) -> np.ndarray:
    if identity not in (PERSISTENT, ALTERNATING):
        raise ValueError("unknown V63 transition identity")
    if not 0.5 < theta < 1.0:
        raise ValueError("V63 theta must be strictly between one half and one")
    transition = np.asarray(anchor["transition"], dtype=np.float64).copy()
    for source, same_side in (
        (LEFT_START, LEFT), (RIGHT_START, RIGHT), (LEFT, LEFT), (RIGHT, RIGHT)
    ):
        other_side = RIGHT if same_side == LEFT else LEFT
        same_probability = theta if identity == PERSISTENT else 1.0 - theta
        transition[LISTEN, source] = 0.0
        transition[LISTEN, source, same_side] = same_probability
        transition[LISTEN, source, other_side] = 1.0 - same_probability
    return transition


def public_episodes(record: dict[str, Any]) -> list[dict[str, Any]]:
    episodes = record.get("episodes")
    if not isinstance(episodes, list) or not episodes:
        raise ValueError("V63 record must contain at least one public episode")
    result = []
    for episode in episodes:
        observations = [int(value) for value in episode["observations"]]
        if not observations or any(value not in (1, 2) for value in observations):
            raise ValueError("V63 all-listen episodes require left/right reports")
        result.append({"observations": observations})
    return result


def exact_filter_episode(
    anchor: dict[str, Any], identity: int, theta: float, observations: Sequence[int]
) -> tuple[float, np.ndarray]:
    transition = family_transition(anchor, identity, theta)[LISTEN]
    observation = np.asarray(anchor["observation"], dtype=np.float64)[LISTEN]
    belief = np.asarray(anchor["initial"], dtype=np.float64).copy()
    log_likelihood = 0.0
    for report in observations:
        weighted = (belief @ transition) * observation[:, int(report)]
        increment = float(weighted.sum())
        if increment <= 0.0 or not math.isfinite(increment):
            return -math.inf, np.zeros_like(belief)
        belief = weighted / increment
        log_likelihood += math.log(increment)
    return log_likelihood, belief


def exact_path(
    anchor: dict[str, Any], identity: int, theta: float, record: dict[str, Any]
) -> tuple[float, np.ndarray]:
    total = 0.0
    current = np.asarray(anchor["initial"], dtype=np.float64)
    for episode in public_episodes(record):
        log_likelihood, current = exact_filter_episode(
            anchor, identity, theta, episode["observations"]
        )
        if not math.isfinite(log_likelihood):
            return -math.inf, np.zeros(5, dtype=np.float64)
        total += log_likelihood
    return total, current


def theta_bin(theta: float, parameter: dict[str, Any], bins: int = 16) -> int:
    low, high = map(float, parameter["support"])
    return min(bins - 1, max(0, int((theta - low) / (high - low) * bins)))


def side_marginal(state: np.ndarray) -> np.ndarray:
    return np.asarray([
        float(state[LEFT_START] + state[LEFT]),
        float(state[RIGHT_START] + state[RIGHT]),
        float(state[TERMINAL]),
    ])


def next_listen_predictive(
    anchor: dict[str, Any], identity: int, theta: float, state: np.ndarray
) -> np.ndarray:
    transition = family_transition(anchor, identity, theta)[LISTEN]
    observation = np.asarray(anchor["observation"], dtype=np.float64)[LISTEN]
    return (state @ transition) @ observation


def exact_inference(anchor: dict[str, Any], record: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    parameter = config["unknownDynamicsFamily"]["continuousParameter"]
    theta, prior_weights = quadrature_rule(
        int(config["exactBenchmark"]["quadratureNodes"]), parameter
    )
    rows: list[dict[str, Any]] = []
    log_masses: list[float] = []
    for identity in (PERSISTENT, ALTERNATING):
        for node_index, (value, prior_weight) in enumerate(zip(theta, prior_weights, strict=True)):
            log_likelihood, state = exact_path(anchor, identity, float(value), record)
            rows.append({
                "identity": identity,
                "node_index": node_index,
                "theta": float(value),
                "state": state,
                "log_likelihood": log_likelihood,
            })
            log_masses.append(-math.log(2.0) + math.log(float(prior_weight)) + log_likelihood)
    weights = normalize_log_weights(log_masses)
    identity_mass = np.zeros(2, dtype=np.float64)
    current_side = np.zeros(3, dtype=np.float64)
    next_observation = np.zeros(4, dtype=np.float64)
    theta_values: list[float] = []
    theta_weights: list[float] = []
    joint_bins: dict[str, float] = {}
    atoms: list[dict[str, Any]] = []
    for row, weight in zip(rows, weights, strict=True):
        if weight <= 0.0:
            continue
        identity = int(row["identity"])
        value = float(row["theta"])
        state = np.asarray(row["state"])
        identity_mass[identity] += weight
        theta_values.append(value)
        theta_weights.append(weight)
        bin_key = f"{identity}:{theta_bin(value, parameter)}"
        joint_bins[bin_key] = joint_bins.get(bin_key, 0.0) + weight
        current_side += weight * side_marginal(state)
        next_observation += weight * next_listen_predictive(anchor, identity, value, state)
        for state_index, probability in enumerate(state):
            if probability > 0.0:
                atoms.append({
                    "identity": identity,
                    "theta": value,
                    "state": state_index,
                    "weight": weight * float(probability),
                })
    atom_total = sum(row["weight"] for row in atoms)
    for atom in atoms:
        atom["weight"] /= atom_total
    return {
        "identity": identity_mass.tolist(),
        "theta_values": theta_values,
        "theta_weights": theta_weights,
        "joint_bins": joint_bins,
        "current_side": current_side.tolist(),
        "next_observation": next_observation.tolist(),
        "log_evidence": logsumexp(log_masses),
        "rows": rows,
        "weights": weights,
        "atoms": atoms,
    }


def systematic_indices(weights: Sequence[float], count: int, seed: int) -> tuple[list[int], float]:
    if len(weights) == 0 or count <= 0:
        raise ValueError("systematic resampling requires positive support and count")
    normalized = np.asarray(weights, dtype=np.float64)
    normalized /= normalized.sum()
    offset = random.Random(seed).random() / count
    positions = offset + np.arange(count, dtype=np.float64) / count
    cumulative = np.cumsum(normalized)
    indices = np.searchsorted(cumulative, positions, side="right")
    indices = np.minimum(indices, len(normalized) - 1)
    return indices.astype(int).tolist(), float(offset)


def particle_filter_episode(
    anchor: dict[str, Any],
    identity: int,
    theta: float,
    observations: Sequence[int],
    budget: int,
    base_seed: int,
    stream_parts: Sequence[Any],
    ess_threshold_fraction: float,
) -> tuple[float, np.ndarray, np.ndarray, dict[str, Any]]:
    initial = np.asarray(anchor["initial"], dtype=np.float64)
    initialization_seed = stable_seed(base_seed, *stream_parts, "initial")
    init_rng = random.Random(initialization_seed)
    cumulative_initial = np.cumsum(initial)
    states = np.asarray([
        int(np.searchsorted(cumulative_initial, init_rng.random(), side="right"))
        for _ in range(budget)
    ], dtype=np.int16)
    weights = np.full(budget, 1.0 / budget, dtype=np.float64)
    observation = np.asarray(anchor["observation"], dtype=np.float64)[LISTEN]
    transition = family_transition(anchor, identity, theta)[LISTEN]
    total_log_likelihood = 0.0
    diagnostics = {
        "resampling_stream_ids": [],
        "resampling_fingerprints": [],
        "ess_fractions": [],
        "resampling_count": 0,
    }
    for tick, report in enumerate(observations):
        transition_seed = stable_seed(base_seed, *stream_parts, tick, "transition")
        transition_rng = random.Random(transition_seed)
        next_states = np.empty_like(states)
        for particle, state in enumerate(states):
            cumulative = np.cumsum(transition[int(state)])
            next_states[particle] = int(
                np.searchsorted(cumulative, transition_rng.random(), side="right")
            )
        states = next_states
        likelihood = observation[states, int(report)]
        increment = float(weights @ likelihood)
        if increment <= 0.0 or not math.isfinite(increment):
            return -math.inf, states, weights, {**diagnostics, "extinct": True}
        total_log_likelihood += math.log(increment)
        weights = weights * likelihood / increment
        ess = 1.0 / float(weights @ weights)
        diagnostics["ess_fractions"].append(ess / budget)
        if ess < ess_threshold_fraction * budget:
            parts = (*stream_parts, tick, "resample")
            resample_seed = stable_seed(base_seed, *parts)
            indices, offset = systematic_indices(weights, budget, resample_seed)
            states = states[np.asarray(indices)]
            weights.fill(1.0 / budget)
            identifier = stream_id(base_seed, *parts)
            diagnostics["resampling_stream_ids"].append(identifier)
            fingerprint_payload = json.dumps(
                {"offset": offset, "indices": indices, "states": states.tolist()},
                sort_keys=True,
                separators=(",", ":"),
            )
            diagnostics["resampling_fingerprints"].append(
                hashlib.sha256(fingerprint_payload.encode()).hexdigest()
            )
            diagnostics["resampling_count"] += 1
    diagnostics["extinct"] = False
    return total_log_likelihood, states, weights, diagnostics


def _full_particle_path(
    anchor: dict[str, Any],
    identity: int,
    theta: float,
    episodes: Sequence[dict[str, Any]],
    budget: int,
    base_seed: int,
    stream_prefix: Sequence[Any],
    ess_threshold_fraction: float,
    purpose: str,
) -> tuple[float, np.ndarray, np.ndarray, list[dict[str, Any]]]:
    total = 0.0
    last_states = np.asarray([], dtype=np.int16)
    last_weights = np.asarray([], dtype=np.float64)
    diagnostics = []
    for episode_index, episode in enumerate(episodes):
        result = particle_filter_episode(
            anchor, identity, theta, episode["observations"], budget, base_seed,
            (*stream_prefix, episode_index, purpose), ess_threshold_fraction,
        )
        log_likelihood, last_states, last_weights, diagnostic = result
        diagnostics.append(diagnostic)
        if not math.isfinite(log_likelihood):
            return -math.inf, last_states, last_weights, diagnostics
        total += log_likelihood
    return total, last_states, last_weights, diagnostics


def _outer_identity_smc(
    anchor: dict[str, Any],
    identity: int,
    record: dict[str, Any],
    config: dict[str, Any],
    outer_budget: int,
    repeat: int,
    population: str,
    *,
    disable_outer_resampling: bool = False,
    likelihood_power: float = 1.0,
) -> dict[str, Any]:
    specification = config["smcSquared"]
    parameter = config["unknownDynamicsFamily"]["continuousParameter"]
    low, high = map(float, parameter["support"])
    alpha, beta = float(parameter["alpha"]), float(parameter["beta"])
    record_id = str(record["id"])
    particles = []
    for slot in range(outer_budget):
        seed = stable_seed(
            int(config["population"]["outerParticleSeed"]),
            population, record_id, identity, outer_budget, repeat, slot, "prior",
        )
        particles.append({
            "theta": scaled_beta_sample(seed, low, high, alpha, beta),
            "weight": 1.0 / outer_budget,
            "log_likelihood": 0.0,
            "states": np.asarray([], dtype=np.int16),
            "state_weights": np.asarray([], dtype=np.float64),
            "ancestor": slot,
        })
    episodes = public_episodes(record)
    diagnostics: dict[str, Any] = {
        "outer_ess_fractions": [],
        "inner_ess_fractions": [],
        "outer_resampling_stream_ids": [],
        "outer_resampling_fingerprints": [],
        "inner_resampling_stream_ids": [],
        "inner_resampling_fingerprints": [],
        "move_attempts": 0,
        "move_accepts": 0,
    }
    log_evidence = 0.0
    resampling_ordinal = 0
    for episode_index, episode in enumerate(episodes):
        log_unnormalized = []
        for slot, particle in enumerate(particles):
            path = particle_filter_episode(
                anchor, identity, float(particle["theta"]), episode["observations"],
                int(specification["innerStateParticleBudget"]),
                int(config["population"]["innerParticleSeed"]),
                (
                    population, record_id, identity, outer_budget, repeat, slot,
                    episode_index, "update",
                ),
                float(specification["innerEssThresholdFraction"]),
            )
            log_likelihood, states, state_weights, inner_diagnostic = path
            diagnostics["inner_ess_fractions"].extend(inner_diagnostic["ess_fractions"])
            diagnostics["inner_resampling_stream_ids"].extend(
                inner_diagnostic["resampling_stream_ids"]
            )
            diagnostics["inner_resampling_fingerprints"].extend(
                inner_diagnostic["resampling_fingerprints"]
            )
            if not math.isfinite(log_likelihood):
                log_unnormalized.append(-math.inf)
                continue
            increment = likelihood_power * log_likelihood
            particle["log_likelihood"] += increment
            particle["states"] = states
            particle["state_weights"] = state_weights
            log_unnormalized.append(math.log(float(particle["weight"])) + increment)
        increment_evidence = logsumexp(log_unnormalized)
        if not math.isfinite(increment_evidence):
            raise RuntimeError("all V63 outer particles became extinct")
        log_evidence += increment_evidence
        normalized = normalize_log_weights(log_unnormalized)
        for particle, weight in zip(particles, normalized, strict=True):
            particle["weight"] = weight
        ess = 1.0 / sum(weight * weight for weight in normalized)
        diagnostics["outer_ess_fractions"].append(ess / outer_budget)
        if (
            not disable_outer_resampling
            and ess < float(specification["outerEssThresholdFraction"]) * outer_budget
        ):
            parts = (
                population, record_id, identity, outer_budget, repeat, episode_index,
                "outer-resample", resampling_ordinal,
            )
            base_seed = int(config["population"]["outerParticleSeed"])
            indices, offset = systematic_indices(
                normalized, outer_budget, stable_seed(base_seed, *parts)
            )
            particles = [copy.deepcopy(particles[index]) for index in indices]
            for slot, particle in enumerate(particles):
                particle["weight"] = 1.0 / outer_budget
                for move in range(int(specification["rejuvenationStepsPerOuterResampling"])):
                    proposal_seed = stable_seed(
                        int(config["population"]["rejuvenationSeed"]),
                        population, record_id, identity, outer_budget, repeat, slot,
                        episode_index, resampling_ordinal, move, "proposal",
                    )
                    proposal_rng = random.Random(proposal_seed)
                    current_theta = float(particle["theta"])
                    proposal_theta = logit_to_theta(
                        theta_to_logit(current_theta, low, high)
                        + proposal_rng.gauss(0.0, float(specification["proposalStandardDeviation"])),
                        low, high,
                    )
                    proposal = _full_particle_path(
                        anchor, identity, proposal_theta, episodes[: episode_index + 1],
                        int(specification["innerStateParticleBudget"]),
                        int(config["population"]["innerParticleSeed"]),
                        (
                            population, record_id, identity, outer_budget, repeat, slot,
                            episode_index, resampling_ordinal, move,
                        ),
                        float(specification["innerEssThresholdFraction"]), "move",
                    )
                    proposed_ll, states, state_weights, proposed_diagnostics = proposal
                    for diagnostic in proposed_diagnostics:
                        diagnostics["inner_ess_fractions"].extend(diagnostic["ess_fractions"])
                        diagnostics["inner_resampling_stream_ids"].extend(
                            diagnostic["resampling_stream_ids"]
                        )
                        diagnostics["inner_resampling_fingerprints"].extend(
                            diagnostic["resampling_fingerprints"]
                        )
                    diagnostics["move_attempts"] += 1
                    if not math.isfinite(proposed_ll):
                        continue
                    proposed_ll *= likelihood_power
                    current_target = (
                        scaled_beta_log_pdf(current_theta, low, high, alpha, beta)
                        + float(particle["log_likelihood"])
                        + log_abs_theta_jacobian(current_theta, low, high)
                    )
                    proposed_target = (
                        scaled_beta_log_pdf(proposal_theta, low, high, alpha, beta)
                        + proposed_ll
                        + log_abs_theta_jacobian(proposal_theta, low, high)
                    )
                    accept_seed = stable_seed(
                        int(config["population"]["rejuvenationSeed"]),
                        population, record_id, identity, outer_budget, repeat, slot,
                        episode_index, resampling_ordinal, move, "accept",
                    )
                    log_uniform = math.log(max(random.Random(accept_seed).random(), 1e-300))
                    if log_uniform < min(0.0, proposed_target - current_target):
                        particle.update({
                            "theta": proposal_theta,
                            "log_likelihood": proposed_ll,
                            "states": states,
                            "state_weights": state_weights,
                        })
                        diagnostics["move_accepts"] += 1
            identifier = stream_id(base_seed, *parts)
            diagnostics["outer_resampling_stream_ids"].append(identifier)
            fingerprint = json.dumps(
                {"offset": offset, "indices": indices, "theta": [p["theta"] for p in particles]},
                sort_keys=True,
                separators=(",", ":"),
            )
            diagnostics["outer_resampling_fingerprints"].append(
                hashlib.sha256(fingerprint.encode()).hexdigest()
            )
            resampling_ordinal += 1
    return {"log_evidence": log_evidence, "particles": particles, "diagnostics": diagnostics}


def _particle_state_distribution(particle: dict[str, Any]) -> np.ndarray:
    result = np.zeros(5, dtype=np.float64)
    states = np.asarray(particle["states"], dtype=int)
    weights = np.asarray(particle["state_weights"], dtype=np.float64)
    for state, weight in zip(states, weights, strict=True):
        result[state] += weight
    return result


def smc2_inference(
    anchor: dict[str, Any],
    record: dict[str, Any],
    config: dict[str, Any],
    outer_budget: int,
    repeat: int,
    population: str,
    *,
    disable_outer_resampling: bool = False,
    likelihood_power: float = 1.0,
) -> dict[str, Any]:
    identity_results = [
        _outer_identity_smc(
            anchor, identity, record, config, outer_budget, repeat, population,
            disable_outer_resampling=disable_outer_resampling,
            likelihood_power=likelihood_power,
        )
        for identity in (PERSISTENT, ALTERNATING)
    ]
    identity_log_mass = [result["log_evidence"] - math.log(2.0) for result in identity_results]
    identity_mass = normalize_log_weights(identity_log_mass)
    parameter = config["unknownDynamicsFamily"]["continuousParameter"]
    theta_values: list[float] = []
    theta_weights: list[float] = []
    joint_bins: dict[str, float] = {}
    current_side = np.zeros(3, dtype=np.float64)
    next_observation = np.zeros(4, dtype=np.float64)
    atoms = []
    for identity, (result, identity_weight) in enumerate(
        zip(identity_results, identity_mass, strict=True)
    ):
        for particle in result["particles"]:
            mass = identity_weight * float(particle["weight"])
            value = float(particle["theta"])
            state = _particle_state_distribution(particle)
            theta_values.append(value)
            theta_weights.append(mass)
            key = f"{identity}:{theta_bin(value, parameter)}"
            joint_bins[key] = joint_bins.get(key, 0.0) + mass
            current_side += mass * side_marginal(state)
            next_observation += mass * next_listen_predictive(anchor, identity, value, state)
            for state_index, probability in enumerate(state):
                if probability > 0.0:
                    atoms.append({
                        "identity": identity,
                        "theta": value,
                        "state": state_index,
                        "weight": mass * float(probability),
                    })
    total = sum(row["weight"] for row in atoms)
    for atom in atoms:
        atom["weight"] /= total
    return {
        "identity": identity_mass,
        "theta_values": theta_values,
        "theta_weights": theta_weights,
        "joint_bins": joint_bins,
        "current_side": current_side.tolist(),
        "next_observation": next_observation.tolist(),
        "log_evidence": logsumexp(identity_log_mass),
        "identity_results": identity_results,
        "atoms": atoms,
    }


def simulate_episode(
    anchor: dict[str, Any], identity: int, theta: float, length: int, seed: int
) -> tuple[list[int], list[int]]:
    rng = random.Random(seed)
    initial = np.asarray(anchor["initial"], dtype=np.float64)
    state = int(np.searchsorted(np.cumsum(initial), rng.random(), side="right"))
    transition = family_transition(anchor, identity, theta)[LISTEN]
    observation = np.asarray(anchor["observation"], dtype=np.float64)[LISTEN]
    reports, states = [], []
    for _ in range(length):
        state = int(np.searchsorted(np.cumsum(transition[state]), rng.random(), side="right"))
        report = int(np.searchsorted(np.cumsum(observation[state]), rng.random(), side="right"))
        states.append(state)
        reports.append(report)
    return reports, states


def sample_weighted_index(weights: Sequence[float], seed: int) -> int:
    normalized = np.asarray(weights, dtype=np.float64)
    normalized /= normalized.sum()
    draw = random.Random(seed).random()
    return int(np.searchsorted(np.cumsum(normalized), draw, side="right"))


def posterior_draws(inference: dict[str, Any], count: int, seed: int) -> list[dict[str, Any]]:
    atoms = inference["atoms"]
    weights = [float(atom["weight"]) for atom in atoms]
    return [
        atoms[sample_weighted_index(weights, stable_seed(seed, draw))].copy()
        for draw in range(count)
    ]


def canonical_map_index(values: Sequence[float], tolerance: float = 1e-12) -> int:
    maximum = max(values)
    return next(index for index, value in enumerate(values) if maximum - value <= tolerance)
