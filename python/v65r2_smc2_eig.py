#!/usr/bin/env python3
"""Targeted exact-zero identity support repair over the frozen V65r1 implementation."""
from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np

import v65_smc2_eig as base
from v22r2_grounding import PROJECT_ROOT
from v64_external_eig import V64Family, action_index, observation_index


# Unchanged V65r1 public operations are re-exported explicitly. The frozen V65r1 source remains
# hash-bound by the V65r2 implementation lock; this module changes only support classification and
# joint identity-measure assembly.
attempted_outcome_leak = base.attempted_outcome_leak
canonicalize_atoms = base.canonicalize_atoms
collapse_map_identity = base.collapse_map_identity
collapse_theta_mean = base.collapse_theta_mean
force_equal_identity_evidence = base.force_equal_identity_evidence
pool_repeats = base.pool_repeats
posterior_summary = base.posterior_summary
rao_blackwellize_measure = base.rao_blackwellize_measure
score_action = base.score_action
score_all_actions = base.score_all_actions
score_state_as_target = base.score_state_as_target
select_action = base.select_action
stable_seed = base.stable_seed


class ImpossiblePublicHistory(RuntimeError):
    """Both frozen identities have exact zero support for a public history."""


class ParticleExtinctionWithPositiveSupport(RuntimeError):
    """Finite particles collapsed although the frozen identity has exact structural support."""


def load_config(path: str | Path = "configs/v65r2-design-lock.json") -> dict[str, Any]:
    value = Path(path)
    if not value.is_absolute():
        value = PROJECT_ROOT / value
    return json.loads(value.read_text())["config_payload"]


def boolean_identity_support(
    family: V64Family, record: dict[str, Any], identity: int
) -> dict[str, Any]:
    """Compute exact topological support; no probability threshold or particle is involved."""
    base._assert_public_record(family, record)
    if identity not in (0, 1):
        raise ValueError("V65r2 identity must be zero or one")
    observation = observation_index(family, record["initial_observation"])
    support = (family.model.initial > 0.0) & (
        family.model.observation[0, :, observation] > 0.0
    )
    state_edge_checks = len(family.model.states)
    if not np.any(support):
        return {
            "supported": False,
            "extinction_tick_zero_based": -1,
            "reachable_state_count": 0,
            "state_edge_checks": state_edge_checks,
            "theta_support_invariant": True,
        }
    theta_support_invariant = True
    for tick, (action_name, observation_name) in enumerate(
        zip(record["actions"], record["observations"], strict=True)
    ):
        action = action_index(family, action_name)
        observation = observation_index(family, observation_name)
        transition_supports = family.transitions[identity, :, action] > 0.0
        invariant = bool(
            np.all(transition_supports == transition_supports[0][None, :, :])
        )
        theta_support_invariant = theta_support_invariant and invariant
        if not invariant:
            raise RuntimeError("V65r2 transition support varies over registered theta nodes")
        state_edge_checks += int(np.count_nonzero(support)) * len(family.model.states)
        successor = np.any(support[:, None] & transition_supports[0], axis=0)
        support = successor & (
            family.model.observation[action, :, observation] > 0.0
        )
        if not np.any(support):
            return {
                "supported": False,
                "extinction_tick_zero_based": tick,
                "reachable_state_count": 0,
                "state_edge_checks": state_edge_checks,
                "theta_support_invariant": theta_support_invariant,
            }
    return {
        "supported": True,
        "extinction_tick_zero_based": None,
        "reachable_state_count": int(np.count_nonzero(support)),
        "state_edge_checks": state_edge_checks,
        "theta_support_invariant": theta_support_invariant,
    }


def classify_particle_extinction(
    exact_support: bool, record_id: str, identity: int
) -> str:
    if not exact_support:
        return "exact_zero_identity_support"
    raise ParticleExtinctionWithPositiveSupport(
        f"V65r2 particle extinction with positive exact support: "
        f"record={record_id} identity={identity}"
    )


def normalize_identity_log_evidence(log_evidence: list[float]) -> np.ndarray:
    if len(log_evidence) != 2:
        raise ValueError("V65r2 requires exactly two identity evidences")
    if not any(math.isfinite(value) for value in log_evidence):
        raise ImpossiblePublicHistory("V65r2 public history is impossible under both identities")
    values = base.normalize_log_weights(
        [value - math.log(2.0) if math.isfinite(value) else -math.inf for value in log_evidence]
    )
    if any(
        not math.isfinite(log_value) and float(weight) != 0.0
        for log_value, weight in zip(log_evidence, values, strict=True)
    ):
        raise RuntimeError("V65r2 assigned posterior mass to exact-zero identity support")
    return values


def _zero_support_result(identity: int, support: dict[str, Any]) -> dict[str, Any]:
    if support["supported"]:
        raise ValueError("V65r2 zero-support result requested for a supported identity")
    return {
        "identity": identity,
        "log_evidence": -math.inf,
        "particles": [],
        "outer_ess_fractions": [],
        "work": base._new_work(),
        "support": support,
        "status": "exact_zero_identity_support",
    }


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
    force_positive_support_particle_extinction_identity: int | None = None,
) -> dict[str, Any]:
    started = time.perf_counter()
    base._assert_public_record(family, record)
    if outer_budget <= 0 or repeat < 0:
        raise ValueError("invalid V65r2 outer budget or repeat")
    support = [boolean_identity_support(family, record, identity) for identity in range(2)]
    if not any(row["supported"] for row in support):
        raise ImpossiblePublicHistory(
            f"V65r2 public history is impossible under both identities: {record['record_id']}"
        )
    tracker = base.StreamTracker()
    identity_results = []
    for identity in range(2):
        if not support[identity]["supported"]:
            identity_results.append(_zero_support_result(identity, support[identity]))
            continue
        if force_positive_support_particle_extinction_identity == identity:
            classify_particle_extinction(True, str(record["record_id"]), identity)
        try:
            result = base._outer_identity_smc(
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
        except RuntimeError as exc:
            if str(exc) != "all V65 outer particles became extinct":
                raise
            classify_particle_extinction(True, str(record["record_id"]), identity)
            raise AssertionError("unreachable after V65r2 positive-support extinction") from exc
        result["support"] = support[identity]
        result["status"] = "positive_exact_support_particle_posterior"
        identity_results.append(result)

    if equal_identity_evidence:
        supported = np.asarray([row["supported"] for row in support], dtype=np.float64)
        identity_mass = supported / supported.sum()
    else:
        identity_mass = normalize_identity_log_evidence(
            [float(row["log_evidence"]) for row in identity_results]
        )
    atoms: list[dict[str, Any]] = []
    work = base._new_work()
    ess_values: list[float] = []
    for identity, (result, identity_weight) in enumerate(
        zip(identity_results, identity_mass, strict=True)
    ):
        base._merge_work(work, result["work"])
        ess_values.extend(result["outer_ess_fractions"])
        if identity_weight <= 0.0:
            if result["particles"]:
                raise RuntimeError("V65r2 exact-zero identity unexpectedly contains particles")
            continue
        for particle in result["particles"]:
            atoms.append(
                {
                    "identity": identity,
                    "theta": float(particle["theta"]),
                    "weight": float(identity_weight) * float(particle["weight"]),
                    "state": base._state_distribution(
                        np.asarray(particle["states"]),
                        np.asarray(particle["state_weights"]),
                        len(family.model.states),
                    ),
                }
            )
    atoms = canonicalize_atoms(atoms)
    work["final_posterior_atom_count"] = len(atoms)
    runtime = time.perf_counter() - started
    identity_summary = posterior_summary(family, atoms)["identity"]
    if any(
        not row["supported"] and float(identity_summary[identity]) != 0.0
        for identity, row in enumerate(support)
    ):
        raise RuntimeError("V65r2 posterior retained mass on an impossible identity")
    return {
        "record_id": record["record_id"],
        "outer_budget": int(outer_budget),
        "repeat": int(repeat),
        "atoms": atoms,
        "identity": identity_summary,
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
            "exact_support_by_identity": [row["supported"] for row in support],
            "support_extinction_tick_by_identity": [
                row["extinction_tick_zero_based"] for row in support
            ],
            "support_state_edge_checks": sum(row["state_edge_checks"] for row in support),
            "exact_zero_identity_count": sum(not row["supported"] for row in support),
            "positive_support_particle_extinction_count": 0,
            "identity_status": [row["status"] for row in identity_results],
        },
    }
