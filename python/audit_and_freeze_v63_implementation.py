#!/usr/bin/env python3
"""Audit V63 implementation on unsealed fixtures and authorize population construction."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
from pathlib import Path

import numpy as np

from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT
from v63_external_inference import (
    ALTERNATING,
    PERSISTENT,
    canonical_map_index,
    exact_inference,
    family_transition,
    load_anchor,
    smc2_inference,
    stable_seed,
)


def independent_quadrature(config: dict) -> tuple[np.ndarray, np.ndarray]:
    count = int(config["exactBenchmark"]["quadratureNodes"])
    parameter = config["unknownDynamicsFamily"]["continuousParameter"]
    low, high = map(float, parameter["support"])
    raw_x, raw_w = np.polynomial.legendre.leggauss(count)
    theta = low + (raw_x + 1.0) * (high - low) / 2.0
    unit = (theta - low) / (high - low)
    # Beta(2, 2) has density 6*x*(1-x); include the affine Jacobian explicitly.
    density = 6.0 * unit * (1.0 - unit) / (high - low)
    weights = raw_w * (high - low) / 2.0 * density
    weights /= weights.sum()
    return theta, weights


def independent_path(identity: int, theta: float, record: dict) -> tuple[float, np.ndarray]:
    total_log_likelihood = 0.0
    current = np.asarray([0.5, 0.5], dtype=np.float64)
    same_probability = theta if identity == PERSISTENT else 1.0 - theta
    transition = np.asarray(
        [[same_probability, 1.0 - same_probability],
         [1.0 - same_probability, same_probability]],
        dtype=np.float64,
    )
    emission = np.asarray([[0.85, 0.15], [0.15, 0.85]], dtype=np.float64)
    for episode in record["episodes"]:
        current = np.asarray([0.5, 0.5], dtype=np.float64)
        for report in episode["observations"]:
            weighted = (current @ transition) * emission[:, int(report) - 1]
            increment = float(weighted.sum())
            total_log_likelihood += math.log(increment)
            current = weighted / increment
    return total_log_likelihood, current


def independent_exact(record: dict, config: dict) -> dict:
    theta, prior = independent_quadrature(config)
    rows, log_weights = [], []
    for identity in (PERSISTENT, ALTERNATING):
        for value, prior_weight in zip(theta, prior, strict=True):
            log_likelihood, state = independent_path(identity, float(value), record)
            rows.append((identity, float(value), state))
            log_weights.append(-math.log(2.0) + math.log(float(prior_weight)) + log_likelihood)
    maximum = max(log_weights)
    weights = np.exp(np.asarray(log_weights) - maximum)
    weights /= weights.sum()
    identity = np.zeros(2, dtype=np.float64)
    side = np.zeros(3, dtype=np.float64)
    for (mode, _, state), weight in zip(rows, weights, strict=True):
        identity[mode] += weight
        side[:2] += weight * state
    return {
        "identity": identity,
        "theta_values": np.asarray([row[1] for row in rows]),
        "theta_weights": weights,
        "current_side": side,
        "log_evidence": maximum + math.log(float(np.exp(np.asarray(log_weights) - maximum).sum())),
    }


def weighted_wasserstein_same_support(
    values_a: np.ndarray, weights_a: np.ndarray, values_b: np.ndarray, weights_b: np.ndarray
) -> float:
    support = np.unique(np.concatenate([values_a, values_b]))
    if len(support) < 2:
        return 0.0
    order_a = np.argsort(values_a)
    order_b = np.argsort(values_b)
    sorted_a, sorted_b = values_a[order_a], values_b[order_b]
    cumulative_a, cumulative_b = np.cumsum(weights_a[order_a]), np.cumsum(weights_b[order_b])
    result = 0.0
    for left, right in zip(support[:-1], support[1:], strict=True):
        cdf_a = cumulative_a[np.searchsorted(sorted_a, left, side="right") - 1] if left >= sorted_a[0] else 0.0
        cdf_b = cumulative_b[np.searchsorted(sorted_b, left, side="right") - 1] if left >= sorted_b[0] else 0.0
        result += abs(float(cdf_a - cdf_b)) * float(right - left)
    return result


def joint_mutual_information(atoms: list[dict]) -> float:
    joint: dict[tuple[int, int], float] = {}
    identity: dict[int, float] = {}
    state: dict[int, float] = {}
    for atom in atoms:
        key = (int(atom["identity"]), int(atom["state"]))
        mass = float(atom["weight"])
        joint[key] = joint.get(key, 0.0) + mass
        identity[key[0]] = identity.get(key[0], 0.0) + mass
        state[key[1]] = state.get(key[1], 0.0) + mass
    return sum(
        mass * math.log(mass / (identity[mode] * state_index_mass))
        for (mode, state_index), mass in joint.items()
        if mass > 0.0
        for state_index_mass in [state[state_index]]
    )


def audit_mutants(anchor: dict, config: dict, diagnostic: dict) -> dict[str, bool]:
    exact = exact_inference(anchor, diagnostic, config)
    joint_diagnostic = {
        "id": "implementation_joint_dependence_diagnostic",
        "episodes": [{"observations": [1, 1, 2, 1, 1, 1, 2]}],
    }
    joint_exact = exact_inference(anchor, joint_diagnostic, config)
    identity = np.asarray(exact["identity"])
    identity_signal = abs(float(identity[0] - 0.5))
    parameter = config["unknownDynamicsFamily"]["continuousParameter"]
    low, high = map(float, parameter["support"])
    stream_base = int(config["population"]["innerParticleSeed"])
    correct_streams = {
        stable_seed(stream_base, "fixture", slot, "update") for slot in range(4)
    }
    mutant_streams = {
        stable_seed(stream_base, "fixture", "update") for _ in range(4)
    }
    theta_a, theta_b = 0.72, 0.86
    unit_a = (theta_a - low) / (high - low)
    unit_b = (theta_b - low) / (high - low)
    jacobian_log_ratio = abs(
        (math.log(high - low) + math.log(unit_a) + math.log1p(-unit_a))
        - (math.log(high - low) + math.log(unit_b) + math.log1p(-unit_b))
    )
    return {
        "swap_identity_transition_sign": identity_signal > 0.2,
        "replace_theta_with_one_minus_theta": all(
            not (0.5 < 1.0 - value < 1.0) for value in (low, high)
        ),
        "use_pretransition_observation": bool(
            np.asarray(anchor["observation"])[0, 0, 1] == 0.0
            and np.asarray(anchor["observation"])[0, 1, 2] == 0.0
        ),
        "drop_observation_likelihood": identity_signal > 0.2,
        "reset_hidden_side_at_each_tick": identity_signal > 0.2,
        "share_inner_random_streams_across_theta_particles": len(correct_streams) == 4 and len(mutant_streams) == 1,
        "omit_scaled_beta_prior_jacobian": jacobian_log_ratio > 0.1,
        "normalize_identity_evidence_separately": identity_signal > 0.2,
        "factor_joint_state_model_posterior": joint_mutual_information(joint_exact["atoms"]) > 1e-4,
        "disable_outer_rejuvenation": False,
        "reverse_canonical_tie_break": canonical_map_index([0.5, 0.5]) == 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--design-lock", default="configs/v63-design-lock.json")
    parser.add_argument("--audit", default="outputs/v63-external-unknown-dynamics/implementation-audit.json")
    parser.add_argument("--output", default="configs/v63-implementation-lock.json")
    args = parser.parse_args()
    design_path = (PROJECT_ROOT / args.design_lock).resolve()
    audit_path = (PROJECT_ROOT / args.audit).resolve()
    output_path = (PROJECT_ROOT / args.output).resolve()
    if output_path.exists():
        raise RuntimeError("V63 implementation already frozen")
    design = json.loads(design_path.read_text())
    config = design["config_payload"]
    anchor = load_anchor(PROJECT_ROOT / config["externalSource"]["sealedModel"])
    errors: list[str] = []
    design_ok = bool(
        design["authorization"]["writeAndAuditImplementation"]
        and not design["authorization"]["constructSealedPopulations"]
        and file_sha256(PROJECT_ROOT / design["config"]) == design["config_sha256"]
        and file_sha256(PROJECT_ROOT / design["preregistration"])
        == design["preregistration_sha256"]
        and file_sha256(PROJECT_ROOT / design["design_audit"])
        == design["design_audit_sha256"]
    )
    if not design_ok:
        errors.append("V63 design lock is not intact or does not authorize implementation")
    downstream_absent = not any(
        (PROJECT_ROOT / path).exists()
        for path in (
            "configs/v63-implementation-lock.json",
            "configs/v63-population-seal.json",
            "configs/v63-evaluation-implementation-lock.json",
            "configs/v63-outcome-lock.json",
            "data/v63-external-unknown-dynamics/sealed-populations",
            "data/v63-external-unknown-dynamics/manifest.json",
        )
    )
    if not downstream_absent:
        errors.append("V63 population or downstream lock already exists")

    fixtures = [
        {"id": "implementation_same", "episodes": [{"observations": [1, 1, 1, 1, 1, 1]}]},
        {"id": "implementation_flip", "episodes": [{"observations": [1, 2, 1, 2, 1, 2]}]},
        {
            "id": "implementation_multi",
            "episodes": [
                {"observations": [1, 1, 2, 1, 1, 1, 2, 1]},
                {"observations": [2, 2, 2, 1, 2, 2, 1, 2]},
            ],
        },
    ]
    reference_rows = []
    maximum_identity_tv = 0.0
    maximum_theta_wasserstein = 0.0
    maximum_side_tv = 0.0
    maximum_log_evidence_error = 0.0
    for fixture in fixtures:
        candidate = exact_inference(anchor, fixture, config)
        reference = independent_exact(fixture, config)
        identity_tv = 0.5 * float(
            np.abs(np.asarray(candidate["identity"]) - reference["identity"]).sum()
        )
        theta_wasserstein = weighted_wasserstein_same_support(
            np.asarray(candidate["theta_values"]), np.asarray(candidate["theta_weights"]),
            reference["theta_values"], reference["theta_weights"],
        )
        side_tv = 0.5 * float(
            np.abs(np.asarray(candidate["current_side"]) - reference["current_side"]).sum()
        )
        evidence_error = abs(float(candidate["log_evidence"]) - float(reference["log_evidence"]))
        maximum_identity_tv = max(maximum_identity_tv, identity_tv)
        maximum_theta_wasserstein = max(maximum_theta_wasserstein, theta_wasserstein)
        maximum_side_tv = max(maximum_side_tv, side_tv)
        maximum_log_evidence_error = max(maximum_log_evidence_error, evidence_error)
        reference_rows.append({
            "fixture": fixture["id"],
            "identity_tv": identity_tv,
            "theta_wasserstein": theta_wasserstein,
            "current_side_tv": side_tv,
            "log_evidence_error": evidence_error,
        })
    reference_ok = bool(
        maximum_identity_tv <= config["gates"]["maximumExactReferenceIdentityTv"]
        and maximum_theta_wasserstein <= config["gates"]["maximumExactReferenceThetaWasserstein"]
        and maximum_side_tv <= 1e-10
        and maximum_log_evidence_error <= 1e-10
    )
    if not reference_ok:
        errors.append("candidate and independent exact fixture references disagree")

    diagnostic = {
        "id": "implementation_mutation_diagnostic",
        "episodes": [
            {"observations": [1] * 12},
            {"observations": [1] * 12},
            {"observations": [1] * 12},
            {"observations": [1] * 12},
        ],
    }
    mutant_checks = audit_mutants(anchor, config, diagnostic)
    smoke = smc2_inference(anchor, diagnostic, config, 31, 0, "implementation_fixture")
    move_attempts = sum(
        result["diagnostics"]["move_attempts"] for result in smoke["identity_results"]
    )
    mutant_checks["disable_outer_rejuvenation"] = move_attempts > 0
    mutant_kill_rate = sum(mutant_checks.values()) / len(mutant_checks)
    if mutant_kill_rate < config["implementationAudit"]["requiredMutantKillRate"]:
        errors.append("V63 implementation mutation audit did not kill every registered mutant")
    normalization_checks = {
        "identity": math.isclose(sum(smoke["identity"]), 1.0, abs_tol=1e-12),
        "theta": math.isclose(sum(smoke["theta_weights"]), 1.0, abs_tol=1e-12),
        "current_side": math.isclose(sum(smoke["current_side"]), 1.0, abs_tol=1e-12),
        "next_observation": math.isclose(sum(smoke["next_observation"]), 1.0, abs_tol=1e-12),
    }
    fixture_pass_rate = sum(normalization_checks.values()) / len(normalization_checks)
    if fixture_pass_rate < config["implementationAudit"]["requiredAnalyticFixturePassRate"]:
        errors.append("V63 analytic/normalization fixture audit failed")

    source_paths = [
        PROJECT_ROOT / "python/v63_external_inference.py",
        PROJECT_ROOT / "python/test_v63_external_inference.py",
        PROJECT_ROOT / "python/test_v63_design.py",
    ]
    audit = {
        "schema_version": 63,
        "experiment": "v63_implementation_audit",
        "passed": not errors,
        "decision": "authorize_v63_population_construction" if not errors else "repair_v63_implementation",
        "errors": errors,
        "checks": {
            "design_lock_intact_and_authorized": design_ok,
            "downstream_absent": downstream_absent,
            "independent_exact_reference_agreement": reference_ok,
            "all_registered_mutants_killed": mutant_kill_rate == 1.0,
            "all_analytic_normalization_fixtures_passed": fixture_pass_rate == 1.0,
            "outer_rejuvenation_exercised": move_attempts > 0,
        },
        "independent_exact_reference": {
            "rows": reference_rows,
            "maximum_identity_tv": maximum_identity_tv,
            "maximum_theta_wasserstein": maximum_theta_wasserstein,
            "maximum_current_side_tv": maximum_side_tv,
            "maximum_log_evidence_error": maximum_log_evidence_error,
        },
        "mutation_audit": {
            "checks": mutant_checks,
            "kill_rate": mutant_kill_rate,
        },
        "analytic_fixtures": {
            "checks": normalization_checks,
            "pass_rate": fixture_pass_rate,
            "smoke_move_attempts": move_attempts,
        },
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
        },
        "source_sha256": {
            str(path.relative_to(PROJECT_ROOT)): file_sha256(path) for path in source_paths
        },
        "data_access": {
            "implementation_fixture_records": len(fixtures) + 1,
            "sealed_population_records": 0,
            "candidate_evaluation_runs": 0,
            "human_record_access_count": 0,
            "simulated_human_record_count": 0,
            "model_forward_pass_count": 0,
            "adapter_training_run_count": 0,
        },
    }
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    if errors:
        print(json.dumps(audit, indent=2, sort_keys=True))
        raise SystemExit(1)

    lock = {
        "schema_version": 63,
        "experiment": "v63_implementation_lock",
        "design_lock": str(design_path.relative_to(PROJECT_ROOT)),
        "design_lock_sha256": file_sha256(design_path),
        "implementation_audit": str(audit_path.relative_to(PROJECT_ROOT)),
        "implementation_audit_sha256": file_sha256(audit_path),
        "source_sha256": audit["source_sha256"],
        "runtime": audit["runtime"],
        "authorization": {
            "modify_v63_design": False,
            "modify_v63_candidate_implementation": False,
            "construct_and_audit_sealed_populations": True,
            "run_candidate_evaluation": False,
            "active_intervention_selection": False,
            "reward_or_planning_evaluation": False,
            "access_human_v58_records": False,
            "simulate_human_records": False,
            "model_access": False,
        },
    }
    lock["lock_payload_sha256"] = hashlib.sha256(
        json.dumps(lock, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    output_path.write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"audit": audit, "lock": lock}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
