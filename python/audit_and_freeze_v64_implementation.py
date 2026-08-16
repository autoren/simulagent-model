#!/usr/bin/env python3
"""Audit, mutation-test, and freeze the exact V64 EIG implementation."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
from pathlib import Path
import subprocess
import sys

import numpy as np

from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT
from v64_external_eig import (
    V64Family,
    assert_public_selection_payload,
    attempted_outcome_leak_selection,
    filter_public_history,
    initial_joint_belief,
    load_family,
    map_identity_belief,
    predict_joint_parameter_observation,
    score_all_actions,
    score_control_policies,
    select_action,
    theta_mean_point_family,
    update_joint_belief,
    wrong_permutation_family,
)
from v64_scalar_reference import (
    atoms_to_dense,
    filter_history as scalar_filter_history,
    load_reference,
    score_all_actions as scalar_score_all_actions,
)


def values(family: V64Family, belief: np.ndarray) -> np.ndarray:
    return np.asarray([row["eig"] for row in score_all_actions(family, belief)])


def mi_from_joint(target: np.ndarray, joint: np.ndarray) -> float:
    predictive = joint.sum(axis=(0, 1))
    result = 0.0
    for identity in range(joint.shape[0]):
        for node in range(joint.shape[1]):
            for observation in range(joint.shape[2]):
                mass = float(joint[identity, node, observation])
                denominator = float(target[identity, node] * predictive[observation])
                if mass > 0.0:
                    result += mass * math.log(mass / denominator)
    return result


def likelihood_squared_values(family: V64Family, belief: np.ndarray) -> np.ndarray:
    target = belief.sum(axis=2)
    result = []
    for action in family.canonical_actions:
        _, joint = predict_joint_parameter_observation(family, belief, action)
        conditional = np.divide(
            joint,
            target[:, :, None],
            out=np.zeros_like(joint),
            where=target[:, :, None] > 0.0,
        )
        squared = conditional * conditional
        squared /= squared.sum(axis=2, keepdims=True)
        pseudo_joint = target[:, :, None] * squared
        result.append(mi_from_joint(target, pseudo_joint))
    return np.asarray(result)


def observe_before_transition_values(family: V64Family, belief: np.ndarray) -> np.ndarray:
    target = belief.sum(axis=2)
    result = []
    for action in family.canonical_actions:
        joint = np.einsum(
            "zqs,so->zqo", belief, family.model.observation[action]
        )
        result.append(mi_from_joint(target, joint))
    return np.asarray(result)


def source_only_family(family: V64Family) -> V64Family:
    transitions = np.broadcast_to(
        family.model.transition[None, None, :, :, :], family.transitions.shape
    ).copy()
    return V64Family(
        model=family.model,
        theta=family.theta.copy(),
        theta_weights=family.theta_weights.copy(),
        static_prior=family.static_prior.copy(),
        transitions=transitions,
        permutations=family.permutations.copy(),
        canonical_actions=family.canonical_actions,
        identity_names=family.identity_names,
        theta_support=family.theta_support,
    )


def public_fixtures(family: V64Family) -> list[dict]:
    fixtures: list[dict] = []
    for initial_observation in family.model.observations:
        try:
            belief, _ = initial_joint_belief(family, initial_observation)
        except ValueError:
            continue
        fixtures.append(
            {
                "initial_observation": initial_observation,
                "actions": [],
                "observations": [],
            }
        )
        for action in family.model.actions:
            _, joint = predict_joint_parameter_observation(family, belief, action)
            for observation, probability in enumerate(joint.sum(axis=(0, 1))):
                if probability <= 1e-10:
                    continue
                fixtures.append(
                    {
                        "initial_observation": initial_observation,
                        "actions": [action],
                        "observations": [family.model.observations[observation]],
                    }
                )
    return fixtures


def mutation_audit(family: V64Family, fixtures: list[dict]) -> dict:
    altered_source = source_only_family(family)
    wrong_permutation = wrong_permutation_family(family)
    maximum_differences = {
        name: 0.0
        for name in (
            "predictive_entropy_target",
            "current_state_target",
            "map_identity_collapse",
            "theta_mean_collapse",
            "wrong_single_action_permutation",
            "source_transition_only",
            "observation_before_transition",
            "factor_static_and_dynamic_belief",
            "likelihood_squared",
        )
    }
    selection_disagreements = {name: 0 for name in maximum_differences}
    reverse_tie_killed = False
    for fixture in fixtures:
        belief, _ = filter_public_history(
            family,
            fixture["initial_observation"],
            fixture["actions"],
            fixture["observations"],
        )
        primary = values(family, belief)
        primary_selected = int(np.argmax(primary))
        controls = score_control_policies(family, belief)
        control_vectors = {
            "predictive_entropy_target": np.asarray(controls["predictive_entropy"]["values"]),
            "current_state_target": np.asarray(controls["state_only_information"]["values"]),
            "map_identity_collapse": np.asarray(controls["map_identity"]["values"]),
            "theta_mean_collapse": np.asarray(controls["theta_mean"]["values"]),
            "wrong_single_action_permutation": values(wrong_permutation, belief),
            "source_transition_only": values(altered_source, belief),
            "observation_before_transition": observe_before_transition_values(family, belief),
            "likelihood_squared": likelihood_squared_values(family, belief),
        }
        static = belief.sum(axis=2)
        state = belief.sum(axis=(0, 1))
        factorized = static[:, :, None] * state[None, None, :]
        factorized /= factorized.sum()
        control_vectors["factor_static_and_dynamic_belief"] = values(family, factorized)
        for name, mutant in control_vectors.items():
            maximum_differences[name] = max(
                maximum_differences[name], float(np.max(np.abs(primary - mutant)))
            )
            if int(np.argmax(mutant)) != primary_selected:
                selection_disagreements[name] += 1
        selection = select_action(family, belief)
        if len(selection["optimal_actions"]) > 1:
            reverse_tie_killed = bool(
                selection["selected"]["action"] != selection["optimal_actions"][-1]
            )
    truth_firewall_killed = False
    outcome_firewall_killed = False
    public = {
        "record_id": "implementation-fixture",
        "prefix_length": 0,
        "initial_observation": "left",
        "actions": [],
        "observations": [],
    }
    try:
        assert_public_selection_payload({**public, "theta": 0.8})
    except PermissionError:
        truth_firewall_killed = True
    try:
        attempted_outcome_leak_selection(public, "right")
    except PermissionError:
        outcome_firewall_killed = True
    killed = {
        name: bool(maximum_differences[name] > 1e-7 or selection_disagreements[name] > 0)
        for name in maximum_differences
    }
    killed.update(
        {
            "omit_west_action": len(family.canonical_actions) == 4,
            "reverse_canonical_tie_break": reverse_tie_killed,
            "truth_field_leakage": truth_firewall_killed,
            "realized_outcome_leakage": outcome_firewall_killed,
        }
    )
    return {
        "killed": killed,
        "killed_count": sum(killed.values()),
        "total_mutants": len(killed),
        "maximum_score_difference": maximum_differences,
        "selection_disagreement_count": selection_disagreements,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--design", default="configs/v64-design-lock.json")
    parser.add_argument(
        "--audit", default="outputs/v64-external-multi-action-eig/implementation-audit.json"
    )
    parser.add_argument("--output", default="configs/v64-implementation-lock.json")
    args = parser.parse_args()
    design_path = (PROJECT_ROOT / args.design).resolve()
    audit_path = (PROJECT_ROOT / args.audit).resolve()
    output_path = (PROJECT_ROOT / args.output).resolve()
    if output_path.exists():
        raise RuntimeError("V64 implementation already frozen")
    design = json.loads(design_path.read_text())
    config = design["config_payload"]
    errors: list[str] = []

    design_ok = bool(
        design["authorization"]["write_and_audit_exact_EIG_implementation"]
        and not design["authorization"]["construct_selection_population"]
        and not design["authorization"]["run_evaluation"]
        and file_sha256(PROJECT_ROOT / design["config"]) == design["config_sha256"]
        and file_sha256(PROJECT_ROOT / design["preregistration"])
        == design["preregistration_sha256"]
        and file_sha256(PROJECT_ROOT / design["design_audit"])
        == design["design_audit_sha256"]
        and file_sha256(PROJECT_ROOT / design["source_v63r1_outcome_lock"])
        == design["source_v63r1_outcome_lock_sha256"]
        and file_sha256(PROJECT_ROOT / design["external_model"])
        == design["external_model_sha256"]
    )
    if not design_ok:
        errors.append("V64 design lock or upstream binding is not intact")

    env = os.environ.copy()
    env["PYTHONPATH"] = str(PROJECT_ROOT / "python")
    tests = subprocess.run(
        [sys.executable, "-m", "unittest", "python/test_v64_external_eig.py"],
        cwd=PROJECT_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    tests_ok = tests.returncode == 0 and "Ran 10 tests" in tests.stderr and "OK" in tests.stderr
    if not tests_ok:
        errors.append("V64 implementation unit tests failed")

    family = load_family(quadrature_nodes=17)
    reference = load_reference(quadrature_nodes=17)
    fixtures = public_fixtures(family)
    max_belief_error = 0.0
    max_evidence_error = 0.0
    max_eig_error = 0.0
    max_predictive_error = 0.0
    selection_membership = []
    for fixture in fixtures:
        candidate_belief, candidate_evidence = filter_public_history(
            family,
            fixture["initial_observation"],
            fixture["actions"],
            fixture["observations"],
        )
        reference_atoms, reference_evidence = scalar_filter_history(
            reference,
            fixture["initial_observation"],
            fixture["actions"],
            fixture["observations"],
        )
        max_belief_error = max(
            max_belief_error,
            float(
                np.max(
                    np.abs(candidate_belief - atoms_to_dense(reference, reference_atoms))
                )
            ),
        )
        max_evidence_error = max(
            max_evidence_error, abs(candidate_evidence - reference_evidence)
        )
        candidate_scores = score_all_actions(family, candidate_belief)
        reference_scores = scalar_score_all_actions(reference, reference_atoms)
        for candidate, scalar in zip(candidate_scores, reference_scores, strict=True):
            max_eig_error = max(max_eig_error, abs(candidate["eig"] - scalar["eig"]))
            max_predictive_error = max(
                max_predictive_error,
                float(
                    np.max(
                        np.abs(
                            np.asarray(candidate["predictive"])
                            - np.asarray(scalar["predictive"])
                        )
                    )
                ),
            )
        maximum = max(row["eig"] for row in reference_scores)
        exact_optimal = {
            row["action"]
            for row in reference_scores
            if row["eig"] >= maximum - config["targetAndObjective"]["tieToleranceNats"]
        }
        selection_membership.append(
            select_action(family, candidate_belief)["selected"]["action"] in exact_optimal
        )
    reference_ok = bool(
        len(fixtures) >= 50
        and max_belief_error <= 2e-14
        and max_evidence_error <= 2e-14
        and max_eig_error <= 2e-13
        and max_predictive_error <= 2e-14
        and all(selection_membership)
    )
    if not reference_ok:
        errors.append("candidate and structurally separate scalar reference disagree")

    mutations = mutation_audit(family, fixtures)
    mutation_ok = bool(
        mutations["total_mutants"] >= 13
        and mutations["killed_count"] == mutations["total_mutants"]
    )
    if not mutation_ok:
        errors.append("V64 mutation audit did not kill every registered mutant")

    sources = [
        PROJECT_ROOT / "python/v64_external_eig.py",
        PROJECT_ROOT / "python/v64_scalar_reference.py",
        PROJECT_ROOT / "python/test_v64_external_eig.py",
        Path(__file__).resolve(),
    ]
    independent_reference_ok = "v64_external_eig" not in (
        PROJECT_ROOT / "python/v64_scalar_reference.py"
    ).read_text()
    if not independent_reference_ok:
        errors.append("V64 scalar reference imports the candidate implementation")

    downstream_absent = not any(
        (PROJECT_ROOT / path).exists()
        for path in (
            "configs/v64-population-seal.json",
            "configs/v64-evaluation-implementation-lock.json",
            "configs/v64-outcome-lock.json",
            "data/v64-external-multi-action-eig",
            "outputs/v64-external-multi-action-eig/evaluation",
        )
    )
    if not downstream_absent:
        errors.append("V64 population or evaluation artifacts exist before implementation lock")

    audit = {
        "schema_version": 64,
        "experiment": "v64_implementation_audit",
        "passed": not errors,
        "decision": "freeze_v64_implementation_and_authorize_population_construction" if not errors else "repair_v64_implementation",
        "errors": errors,
        "checks": {
            "design_and_upstream_bindings": design_ok,
            "unit_tests": tests_ok,
            "candidate_scalar_reference_agreement": reference_ok,
            "independent_reference_source": independent_reference_ok,
            "all_mutants_killed": mutation_ok,
            "downstream_absent": downstream_absent,
        },
        "unit_test_output": {"stdout": tests.stdout, "stderr": tests.stderr},
        "reference_agreement": {
            "fixtures": len(fixtures),
            "maximum_joint_belief_error": max_belief_error,
            "maximum_log_evidence_error": max_evidence_error,
            "maximum_candidate_eig_error": max_eig_error,
            "maximum_predictive_probability_error": max_predictive_error,
            "optimal_set_membership_rate": float(np.mean(selection_membership)),
        },
        "mutation_audit": mutations,
        "runtime": {"python": platform.python_version(), "numpy": np.__version__},
        "source_sha256": {
            str(path.relative_to(PROJECT_ROOT)): file_sha256(path) for path in sources
        },
        "data_access": {
            "implementation_fixture_records": len(fixtures),
            "v64_evaluation_population_records": 0,
            "v64_candidate_evaluation_runs": 0,
            "v64_SBC_runs": 0,
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
        "schema_version": 64,
        "experiment": "v64_implementation_lock",
        "design_lock": str(design_path.relative_to(PROJECT_ROOT)),
        "design_lock_sha256": file_sha256(design_path),
        "implementation_audit": str(audit_path.relative_to(PROJECT_ROOT)),
        "implementation_audit_sha256": file_sha256(audit_path),
        "source_sha256": audit["source_sha256"],
        "runtime": audit["runtime"],
        "authorization": {
            "modify_v64_design": False,
            "modify_v64_exact_implementation": False,
            "construct_and_audit_sealed_populations": True,
            "run_v64_evaluation": False,
            "approximate_particle_acquisition": False,
            "reward_planning": False,
            "formal_verification": False,
            "access_human_data": False,
            "simulate_human_data": False,
            "model_access": False,
            "adapter_training": False,
        },
    }
    lock["lock_payload_sha256"] = hashlib.sha256(
        json.dumps(lock, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    output_path.write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"audit": audit, "lock": lock}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
