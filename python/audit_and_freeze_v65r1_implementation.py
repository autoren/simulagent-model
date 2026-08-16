#!/usr/bin/env python3
"""Audit and freeze the repaired V65r1 SMC² and EIG implementation."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import platform
from pathlib import Path

import numpy as np

from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT
from v62_external_pomdp import POMDPModel
from v64_external_eig import V64Family, load_family
from v65_scalar_reference import (
    scalar_rao_blackwellize_measure,
    scalar_score_all_actions,
    scalar_state_as_target,
    scalar_summary,
)
from v65_smc2_eig import (
    attempted_outcome_leak,
    canonicalize_atoms,
    load_config,
    pool_repeats,
    posterior_summary,
    rao_blackwellize_measure,
    score_action,
    score_all_actions,
    score_state_as_target,
    select_action,
    smc2_inference,
)


def _fixture_atoms(family: V64Family, seed: int, count: int = 9) -> list[dict]:
    rng = np.random.default_rng(seed)
    atoms = []
    for index in range(count):
        state = rng.dirichlet(np.linspace(0.7, 1.7, len(family.model.states)))
        atoms.append(
            {
                "identity": index % 2,
                "theta": float(0.61 + 0.33 * rng.random()),
                "weight": float(rng.random() + 0.05),
                "state": state,
            }
        )
    total = sum(row["weight"] for row in atoms)
    for atom in atoms:
        atom["weight"] /= total
    return atoms


def _action_dependent_family(family: V64Family) -> V64Family:
    observation = family.model.observation.copy()
    east = family.model.actions.index("e")
    observation[east] = np.roll(observation[east], 1, axis=1)
    model = POMDPModel(
        name="v65-action-dependent-observation-fixture",
        states=family.model.states,
        actions=family.model.actions,
        observations=family.model.observations,
        discount=family.model.discount,
        initial=family.model.initial.copy(),
        transition=family.model.transition.copy(),
        observation=observation,
        reward=family.model.reward.copy(),
    )
    return V64Family(
        model=model,
        theta=family.theta.copy(),
        theta_weights=family.theta_weights.copy(),
        static_prior=family.static_prior.copy(),
        transitions=family.transitions.copy(),
        permutations=family.permutations.copy(),
        canonical_actions=family.canonical_actions,
        identity_names=family.identity_names,
        theta_support=family.theta_support,
    )


def _entropy(values: list[float]) -> float:
    return -sum(value * math.log(value) for value in values if value > 0.0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--design-lock", default="configs/v65r1-design-lock.json")
    parser.add_argument(
        "--audit", default="outputs/v65r1-nested-predictive-repair/implementation-audit.json"
    )
    parser.add_argument("--output", default="configs/v65r1-implementation-lock.json")
    args = parser.parse_args()

    design_path = (PROJECT_ROOT / args.design_lock).resolve()
    audit_path = (PROJECT_ROOT / args.audit).resolve()
    output_path = (PROJECT_ROOT / args.output).resolve()
    if output_path.exists():
        raise RuntimeError("V65r1 implementation already frozen")
    design = json.loads(design_path.read_text())
    config = load_config(design_path)
    errors: list[str] = []

    design_ok = bool(
        design["authorization"]["write_and_audit_repaired_implementation"]
        and not design["authorization"]["materialize_subset"]
        and not design["authorization"]["run_evaluation"]
        and not design["authorization"]["reward_planning"]
        and file_sha256(PROJECT_ROOT / design["repair"]) == design["repair_sha256"]
        and file_sha256(PROJECT_ROOT / design["preregistration"])
        == design["preregistration_sha256"]
        and file_sha256(PROJECT_ROOT / design["design_audit"])
        == design["design_audit_sha256"]
        and config["approximateAcquisition"]["raoBlackwellizeKnownConditionalState"]
        and not config["approximateAcquisition"][
            "integrateInnerStateBeforeTreatingAnOuterParticleAsAStaticLatentAtom"
        ]
    )
    if not design_ok:
        errors.append("V65r1 design authorization, repair binding, or effective config is invalid")

    downstream = (
        "configs/v65r1-subset-seal.json",
        "configs/v65r1-evaluation-implementation-lock.json",
        "configs/v65r1-outcome-lock.json",
        "data/v65-smc2-eig-portability",
        "outputs/v65r1-nested-predictive-repair/evaluation",
    )
    downstream_absent = not any((PROJECT_ROOT / path).exists() for path in downstream)
    if not downstream_absent:
        errors.append("V65r1 subset or evaluation artifacts exist before implementation lock")

    family = load_family(quadrature_nodes=31)
    rng_seed = int(config["seeds"]["fixtureSeed"])
    histories = [
        {
            "record_id": "v65r1-implementation-root",
            "prefix_length": 0,
            "initial_observation": "left",
            "actions": [],
            "observations": [],
        },
        {
            "record_id": "v65r1-implementation-one",
            "prefix_length": 1,
            "initial_observation": "right",
            "actions": ["n"],
            "observations": ["right"],
        },
        {
            "record_id": "v65r1-implementation-two",
            "prefix_length": 2,
            "initial_observation": "left",
            "actions": ["e", "n"],
            "observations": ["neither", "left"],
        },
    ]
    fixture_rows = []
    max_eig_error = 0.0
    max_predictive_error = 0.0
    max_summary_error = 0.0
    max_rb_state_error = 0.0
    max_state_target_error = 0.0
    for fixture in range(14):
        atoms = _fixture_atoms(family, rng_seed + fixture)
        thirds = [atoms[index::3] for index in range(3)]
        repeats = []
        for repeat, rows in enumerate(thirds):
            normalized = canonicalize_atoms(rows)
            repeats.append(
                {
                    "record_id": f"fixture-{fixture}",
                    "outer_budget": 7,
                    "repeat": repeat,
                    "normalizes": True,
                    "atoms": normalized,
                }
            )
        pooled = pool_repeats(repeats)
        history = histories[fixture % len(histories)]
        history = {**history, "record_id": f"fixture-{fixture}"}
        candidate_rb = rao_blackwellize_measure(family, pooled, history)
        scalar_rb = scalar_rao_blackwellize_measure(family, pooled, history)
        candidate_summary = posterior_summary(family, pooled)
        scalar_posterior = scalar_summary(family, pooled)
        summary_error = max(
            float(np.max(np.abs(candidate_summary["identity"] - scalar_posterior["identity"]))),
            float(np.max(np.abs(candidate_summary["state"] - scalar_posterior["state"]))),
            float(np.max(np.abs(candidate_summary["joint_bins"] - scalar_posterior["joint_bins"]))),
        )
        max_summary_error = max(max_summary_error, summary_error)
        candidate_rb_atoms = canonicalize_atoms(candidate_rb["atoms"])
        scalar_rb_atoms = canonicalize_atoms(scalar_rb["atoms"])
        rb_state_error = max(
            float(np.max(np.abs(left["state"] - right["state"])))
            for left, right in zip(candidate_rb_atoms, scalar_rb_atoms, strict=True)
        )
        max_rb_state_error = max(max_rb_state_error, rb_state_error)
        candidate_scores = score_all_actions(family, candidate_rb)
        scalar_scores = scalar_score_all_actions(family, scalar_rb)
        for left, right in zip(candidate_scores, scalar_scores, strict=True):
            max_eig_error = max(max_eig_error, abs(left["eig"] - right["eig"]))
            max_predictive_error = max(
                max_predictive_error,
                float(np.max(np.abs(np.asarray(left["predictive"]) - right["predictive"]))),
            )
        for action in family.canonical_actions:
            max_state_target_error = max(
                max_state_target_error,
                abs(
                    score_state_as_target(family, pooled, action)
                    - scalar_state_as_target(family, pooled, action)
                ),
            )
        fixture_rows.append(
            {
                "fixture": fixture,
                "history_length": history["prefix_length"],
                "candidate_selection": select_action(family, candidate_rb)["selected"]["action"],
                "scalar_selection": scalar_scores[
                    int(np.argmax([row["eig"] for row in scalar_scores]))
                ]["action"],
                "summary_error": summary_error,
                "rb_state_error": rb_state_error,
            }
        )
    reference_ok = bool(
        max_eig_error <= 1e-12
        and max_predictive_error <= 1e-12
        and max_summary_error <= 1e-12
        and max_rb_state_error <= 1e-12
        and max_state_target_error <= 1e-12
        and all(row["candidate_selection"] == row["scalar_selection"] for row in fixture_rows)
    )
    if not reference_ok:
        errors.append("candidate posterior, Rao-Blackwellized EIG, or scalar reference disagrees")

    tiny_config = copy.deepcopy(config)
    tiny_config["smcSquared"]["innerStateParticleBudget"] = 15
    algorithm_history = {
        "record_id": "audit-search-2-0-2-left",
        "prefix_length": 2,
        "initial_observation": "left",
        "actions": ["n", "n"],
        "observations": ["left", "neither"],
    }
    repeats = [
        smc2_inference(family, algorithm_history, tiny_config, 7, repeat)
        for repeat in range(3)
    ]
    deterministic = smc2_inference(family, algorithm_history, tiny_config, 7, 0)
    pooled = pool_repeats(repeats)
    repaired = rao_blackwellize_measure(family, pooled, algorithm_history)
    algorithm_ok = bool(
        all(row["normalizes"] for row in repeats)
        and pooled["normalizes"]
        and posterior_summary(family, pooled)["normalizes"]
        and len(select_action(family, repaired)["scores"]) == 4
        and [row["theta"] for row in repeats[0]["atoms"]]
        == [row["theta"] for row in deterministic["atoms"]]
        and np.allclose(
            [row["weight"] for row in repeats[0]["atoms"]],
            [row["weight"] for row in deterministic["atoms"]],
            atol=0.0,
            rtol=0.0,
        )
        and sum(row["diagnostics"]["work"]["outer_resampling_count"] for row in repeats) > 0
        and sum(row["diagnostics"]["work"]["pmmh_attempt_count"] for row in repeats) > 0
        and all(row["diagnostics"]["random_stream_collision_count"] == 0 for row in repeats)
    )
    if not algorithm_ok:
        errors.append("tiny sequential SMC2 fixture failed normalization, determinism, or work checks")

    base_scores = np.asarray([row["eig"] for row in score_all_actions(family, repaired)])
    plugin_scores = np.asarray([row["eig"] for row in score_all_actions(family, pooled)])
    separate_scores = np.mean(
        [
            [
                row["eig"]
                for row in score_all_actions(
                    family,
                    rao_blackwellize_measure(
                        family, repeat, algorithm_history, allow_unpooled_fixture=True
                    ),
                )
            ]
            for repeat in repeats
        ],
        axis=0,
    )
    first_scores = np.asarray(
        [
            row["eig"]
            for row in score_all_actions(
                family,
                rao_blackwellize_measure(
                    family, repeats[0], algorithm_history, allow_unpooled_fixture=True
                ),
            )
        ]
    )
    omitted_reset = pool_repeats(
        [
            smc2_inference(
                family,
                algorithm_history,
                tiny_config,
                7,
                repeat,
                omit_reset_observation=True,
            )
            for repeat in range(3)
        ]
    )
    omitted_likelihood = pool_repeats(
        [
            smc2_inference(
                family,
                algorithm_history,
                tiny_config,
                7,
                repeat,
                likelihood_power=0.0,
            )
            for repeat in range(3)
        ]
    )
    equal_identity = pool_repeats(
        [
            smc2_inference(
                family,
                algorithm_history,
                tiny_config,
                7,
                repeat,
                equal_identity_evidence=True,
            )
            for repeat in range(3)
        ]
    )
    wrong_permutation_scores = np.asarray(
        [
            score_action(family, repaired, action, wrong_permutation=True)["eig"]
            for action in family.canonical_actions
        ]
    )
    state_target = np.asarray(
        [score_state_as_target(family, pooled, action) for action in family.canonical_actions]
    )
    predictive_entropy = np.asarray(
        [_entropy(row["predictive"]) for row in score_all_actions(family, repaired)]
    )
    no_resampling = smc2_inference(
        family,
        algorithm_history,
        tiny_config,
        7,
        0,
        disable_outer_resampling=True,
    )
    no_rejuvenation = smc2_inference(
        family,
        algorithm_history,
        tiny_config,
        7,
        0,
        disable_rejuvenation=True,
    )
    shared_stream_failure = False
    shared_stream_collisions = 0
    try:
        shared = smc2_inference(
            family,
            algorithm_history,
            tiny_config,
            7,
            0,
            shared_inner_stream=True,
        )
        shared_stream_collisions = int(
            shared["diagnostics"]["random_stream_collision_count"]
        )
    except RuntimeError:
        shared_stream_failure = True
    action_dependent = _action_dependent_family(family)
    correct_observation = score_action(
        action_dependent, repaired, action_dependent.model.actions.index("e")
    )
    wrong_observation = score_action(
        action_dependent,
        repaired,
        action_dependent.model.actions.index("e"),
        observation_action_override=action_dependent.model.actions.index("n"),
    )
    leak_rejected = False
    try:
        attempted_outcome_leak(algorithm_history, "future")
    except PermissionError:
        leak_rejected = True

    mutants = {
        "omit_reset_conditioning": float(
            np.max(
                np.abs(
                    posterior_summary(family, pooled)["state"]
                    - posterior_summary(family, omitted_reset)["state"]
                )
            )
        )
        > 1e-4,
        "wrong_actuator_permutation": float(np.max(np.abs(base_scores - wrong_permutation_scores)))
        > 1e-6,
        "omit_observation_likelihood": float(
            np.max(
                np.abs(
                    posterior_summary(family, pooled)["identity"]
                    - posterior_summary(family, omitted_likelihood)["identity"]
                )
            )
        )
        > 1e-4,
        "force_equal_identity_evidence": float(
            np.max(
                np.abs(
                    posterior_summary(family, pooled)["identity"]
                    - posterior_summary(family, equal_identity)["identity"]
                )
            )
        )
        > 1e-4,
        "state_as_information_target": float(np.max(np.abs(base_scores - state_target))) > 1e-6,
        "predictive_entropy_instead_of_EIG": float(
            np.max(np.abs(base_scores - predictive_entropy))
        )
        > 1e-6,
        "score_then_average_repeats": float(np.max(np.abs(base_scores - separate_scores)))
        > 1e-8,
        "first_repeat_only": float(np.max(np.abs(base_scores - first_scores))) > 1e-8,
        "disable_outer_resampling": (
            repeats[0]["diagnostics"]["work"]["outer_resampling_count"] > 0
            and no_resampling["diagnostics"]["work"]["outer_resampling_count"] == 0
        ),
        "disable_outer_rejuvenation": (
            repeats[0]["diagnostics"]["work"]["pmmh_attempt_count"] > 0
            and no_rejuvenation["diagnostics"]["work"]["pmmh_attempt_count"] == 0
        ),
        "share_inner_streams_across_outer_particles": (
            shared_stream_failure or shared_stream_collisions > 0
        ),
        "wrong_observation_action_index": float(
            np.max(
                np.abs(
                    np.asarray(correct_observation["predictive"])
                    - np.asarray(wrong_observation["predictive"])
                )
            )
        )
        > 1e-6,
        "omit_one_candidate_action": len(score_all_actions(family, repaired)[:-1]) != 4,
        "plugin_inner_particle_state_predictive": float(
            np.max(np.abs(base_scores - plugin_scores))
        )
        > 1e-6,
        "outcome_leakage": leak_rejected,
    }
    mutation_rate = sum(mutants.values()) / len(mutants)
    mutation_ok = mutation_rate == 1.0
    if not mutation_ok:
        errors.append("one or more V65r1 implementation mutants survived")

    analytic = {
        "transition_rows_normalized": bool(
            np.max(np.abs(family.transitions.sum(axis=-1) - 1.0)) <= 1e-12
        ),
        "duplicate_static_atoms_merged": len(
            canonicalize_atoms(
                [
                    {**pooled["atoms"][0], "weight": 0.4},
                    {**pooled["atoms"][0], "weight": 0.6},
                ]
            )
        )
        == 1,
        "point_mass_static_latent_zero_EIG": max(
            abs(row["eig"])
            for row in score_all_actions(
                family,
                {
                    "atoms": [
                        {
                            **repaired["atoms"][0],
                            "weight": 1.0,
                        }
                    ]
                },
            )
        )
        <= 1e-12,
        "pool_before_score_is_noncommutative": float(
            np.max(np.abs(base_scores - separate_scores))
        )
        > 1e-8,
        "Rao_Blackwellization_preserves_static_weights": np.allclose(
            [row["weight"] for row in pooled["atoms"]],
            [row["weight"] for row in repaired["atoms"]],
            atol=1e-15,
            rtol=0.0,
        ),
        "all_four_named_candidates_scored": [
            row["action"] for row in score_all_actions(family, repaired)
        ]
        == ["n", "e", "s", "w"],
    }
    analytic_rate = sum(analytic.values()) / len(analytic)
    analytic_ok = analytic_rate == 1.0
    if not analytic_ok:
        errors.append("one or more V65r1 analytic fixtures failed")

    checks = {
        "v65r1_design_authorization_and_bindings": design_ok,
        "downstream_absent": downstream_absent,
        "independent_scalar_posterior_predictive_EIG_and_Rao_Blackwellization": reference_ok,
        "sequential_SMC2_normalization_determinism_streams_and_work": algorithm_ok,
        "all_registered_mutants_killed": mutation_ok,
        "all_analytic_fixtures_passed": analytic_ok,
    }
    audit = {
        "schema_version": "65r1",
        "experiment": "v65r1_implementation_audit",
        "passed": not errors and all(checks.values()),
        "decision": (
            "freeze_v65r1_implementation_and_authorize_subset_materialization_only"
            if not errors and all(checks.values())
            else "repair_v65r1_implementation"
        ),
        "errors": errors,
        "checks": checks,
        "reference_agreement": {
            "fixtures": len(fixture_rows),
            "candidate_action_comparisons": len(fixture_rows) * 4,
            "maximum_EIG_error": max_eig_error,
            "maximum_predictive_probability_error": max_predictive_error,
            "maximum_posterior_summary_error": max_summary_error,
            "maximum_Rao_Blackwellized_state_error": max_rb_state_error,
            "maximum_state_target_information_error": max_state_target_error,
            "fixture_rows": fixture_rows,
        },
        "algorithm_fixture": {
            "history": algorithm_history,
            "outer_budget": 7,
            "inner_budget": tiny_config["smcSquared"]["innerStateParticleBudget"],
            "repeats": 3,
            "work_by_repeat": [row["diagnostics"]["work"] for row in repeats],
            "stream_collisions_by_repeat": [
                row["diagnostics"]["random_stream_collision_count"] for row in repeats
            ],
        },
        "mutation_audit": {
            "checks": mutants,
            "killed": sum(mutants.values()),
            "registered": len(mutants),
            "kill_rate": mutation_rate,
        },
        "analytic_fixtures": {
            "checks": analytic,
            "passed": sum(analytic.values()),
            "registered": len(analytic),
            "pass_rate": analytic_rate,
        },
        "data_access": {
            "v64_selection_public_records_loaded": 0,
            "v64_selection_audit_records_loaded": 0,
            "v65_subset_records_materialized": 0,
            "candidate_evaluation_runs": 0,
            "human_records": 0,
            "model_forward_passes": 0,
            "adapter_training_runs": 0,
        },
    }
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    if not audit["passed"]:
        print(json.dumps(audit, indent=2, sort_keys=True))
        raise SystemExit(1)

    source_files = (
        "python/v65_smc2_eig.py",
        "python/v65_scalar_reference.py",
        "python/test_v65_smc2_eig.py",
        "python/audit_and_freeze_v65r1_implementation.py",
    )
    lock = {
        "schema_version": "65r1",
        "experiment": "v65r1_implementation_lock",
        "design_lock": str(design_path.relative_to(PROJECT_ROOT)),
        "design_lock_sha256": file_sha256(design_path),
        "implementation_audit": str(audit_path.relative_to(PROJECT_ROOT)),
        "implementation_audit_sha256": file_sha256(audit_path),
        "source_sha256": {
            path: file_sha256(PROJECT_ROOT / path) for path in source_files
        },
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
        },
        "authorization": {
            "modify_v65_or_v65r1_design": False,
            "modify_v65r1_implementation": False,
            "materialize_and_audit_subset": True,
            "write_evaluator": False,
            "run_evaluation": False,
            "reward_planning": False,
            "formal_verification": False,
            "human_data": False,
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
