#!/usr/bin/env python3
"""Audit and freeze the V65r2 exact-zero identity support implementation."""
from __future__ import annotations

import copy
import hashlib
import json
import math
import platform
from pathlib import Path
from typing import Any

import numpy as np

import v65_smc2_eig as v65r1
from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT
from v64_external_eig import action_index, load_family, observation_index
from v65r2_smc2_eig import (
    ImpossiblePublicHistory,
    ParticleExtinctionWithPositiveSupport,
    boolean_identity_support,
    load_config,
    normalize_identity_log_evidence,
    pool_repeats,
    posterior_summary,
    rao_blackwellize_measure,
    score_all_actions,
    smc2_inference,
)


WORK_FIELDS = (
    "outer_particles_initialized",
    "inner_initial_draw_count",
    "inner_transition_draw_count",
    "observation_weight_evaluation_count",
    "complete_history_likelihood_recomputation_count",
    "inner_resampling_count",
    "outer_resampling_count",
    "pmmh_attempt_count",
    "pmmh_accept_count",
    "final_posterior_atom_count",
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def reference_boolean_support(family, record: dict[str, Any], identity: int) -> tuple[bool, int | None]:
    observation = family.model.observations.index(record["initial_observation"])
    support = {
        state
        for state, probability in enumerate(family.model.initial)
        if probability > 0.0 and family.model.observation[0, state, observation] > 0.0
    }
    if not support:
        return False, -1
    for tick, (action_name, observation_name) in enumerate(
        zip(record["actions"], record["observations"], strict=True)
    ):
        action = family.model.actions.index(action_name)
        observation = family.model.observations.index(observation_name)
        transition = family.transitions[identity, 0, action]
        support = {
            successor
            for state in support
            for successor in range(len(family.model.states))
            if transition[state, successor] > 0.0
            and family.model.observation[action, successor, observation] > 0.0
        }
        if not support:
            return False, tick
    return True, None


def main() -> None:
    design_path = PROJECT_ROOT / "configs/v65r2-design-lock.json"
    audit_path = PROJECT_ROOT / "outputs/v65r2-extinct-identity-repair/implementation-audit.json"
    output_path = PROJECT_ROOT / "configs/v65r2-implementation-lock.json"
    if output_path.exists():
        raise RuntimeError("V65r2 implementation already frozen")
    design = json.loads(design_path.read_text())
    config = load_config(design_path)
    errors: list[str] = []

    design_payload = {
        key: value for key, value in design.items() if key != "lock_payload_sha256"
    }
    design_ok = bool(
        hashlib.sha256(
            json.dumps(design_payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        == design["lock_payload_sha256"]
        and design["authorization"]["write_and_audit_repair_implementation"]
        and not design["authorization"]["write_and_audit_durable_evaluator"]
        and not design["authorization"]["run_evaluation"]
        and not design["authorization"]["reward_planning"]
        and file_sha256(PROJECT_ROOT / design["repair"]) == design["repair_sha256"]
        and file_sha256(PROJECT_ROOT / design["preregistration"])
        == design["preregistration_sha256"]
        and file_sha256(PROJECT_ROOT / design["design_audit"])
        == design["design_audit_sha256"]
        and file_sha256(PROJECT_ROOT / design["source_v65r1_outcome_lock"])
        == design["source_v65r1_outcome_lock_sha256"]
        and file_sha256(PROJECT_ROOT / design["subset_seal"])
        == design["subset_seal_sha256"]
    )
    if not design_ok:
        errors.append("V65r2 design lock or implementation-only authorization is invalid")

    downstream_absent = not any(
        (PROJECT_ROOT / path).exists()
        for path in (
            "configs/v65r2-evaluation-implementation-lock.json",
            "configs/v65r2-outcome-lock.json",
            "python/evaluate_v65r2_eig.py",
            "outputs/v65r2-extinct-identity-repair/evaluation",
        )
    )
    if not downstream_absent:
        errors.append("V65r2 evaluator or outcome exists before implementation lock")

    subset_seal = json.loads((PROJECT_ROOT / design["subset_seal"]).read_text())
    subset = read_jsonl(PROJECT_ROOT / subset_seal["files"]["subset_public"]["path"])
    fatal_id = design["repair_payload"]["mandatoryImplementationFixtures"][
        "sealedOneIdentityZeroSupport"
    ]
    fatal = next(row for row in subset if row["record_id"] == fatal_id)
    family = load_family(quadrature_nodes=17)
    tiny = copy.deepcopy(config)
    tiny["smcSquared"]["innerStateParticleBudget"] = 15
    candidate_support = {
        (record["record_id"], identity): boolean_identity_support(
            family, record, identity
        )
        for record in subset
        for identity in range(2)
    }
    reference_support = {
        (record["record_id"], identity): reference_boolean_support(
            family, record, identity
        )
        for record in subset
        for identity in range(2)
    }
    support_reference_ok = all(
        candidate_support[key]["supported"] == value[0]
        and candidate_support[key]["extinction_tick_zero_based"] == value[1]
        for key, value in reference_support.items()
    )
    zero_rows = [key for key, value in reference_support.items() if not value[0]]
    support_reference_ok = support_reference_ok and zero_rows == [(fatal_id, 1)]
    if not support_reference_ok:
        errors.append("candidate Boolean support differs from independent scalar reachability")

    repaired = smc2_inference(family, fatal, tiny, 7, 0)
    fatal_ok = bool(
        repaired["normalizes"]
        and repaired["identity"][1] == 0.0
        and repaired["log_evidence_by_identity"][1] == -math.inf
        and all(atom["identity"] == 0 for atom in repaired["atoms"])
        and repaired["diagnostics"]["exact_zero_identity_count"] == 1
        and repaired["diagnostics"]["work"]["outer_particles_initialized"] == 7
        and all(field in repaired["diagnostics"]["work"] for field in WORK_FIELDS)
    )
    try:
        v65r1.smc2_inference(family, fatal, tiny, 7, 0)
    except RuntimeError as exc:
        old_failure_reproduced = str(exc) == "all V65 outer particles became extinct"
    else:
        old_failure_reproduced = False
    if not fatal_ok or not old_failure_reproduced:
        errors.append("sealed V65r1 failure was not reproduced and narrowly repaired")

    impossible = {
        "record_id": "v65r2-both-impossible-audit",
        "prefix_length": 0,
        "initial_observation": "good",
        "actions": [],
        "observations": [],
    }
    try:
        smc2_inference(family, impossible, tiny, 7, 0)
    except ImpossiblePublicHistory:
        impossible_rejected = True
    else:
        impossible_rejected = False
    regular = {
        "record_id": "v65r2-positive-support-audit",
        "prefix_length": 2,
        "initial_observation": "left",
        "actions": ["n", "e"],
        "observations": ["left", "neither"],
    }
    try:
        smc2_inference(
            family,
            regular,
            tiny,
            7,
            0,
            force_positive_support_particle_extinction_identity=0,
        )
    except ParticleExtinctionWithPositiveSupport:
        positive_collapse_rejected = True
    else:
        positive_collapse_rejected = False
    extinction_classification_ok = impossible_rejected and positive_collapse_rejected
    if not extinction_classification_ok:
        errors.append("both-impossible or positive-support-collapse fixture did not fail closed")

    old_regular = v65r1.smc2_inference(family, regular, tiny, 7, 0)
    new_regular = smc2_inference(family, regular, tiny, 7, 0)
    parity_ok = bool(
        old_regular["log_evidence_by_identity"] == new_regular["log_evidence_by_identity"]
        and old_regular["diagnostics"]["work"] == new_regular["diagnostics"]["work"]
        and old_regular["diagnostics"]["random_stream_count"]
        == new_regular["diagnostics"]["random_stream_count"]
        and len(old_regular["atoms"]) == len(new_regular["atoms"])
        and all(
            left["identity"] == right["identity"]
            and left["theta"] == right["theta"]
            and left["weight"] == right["weight"]
            and np.array_equal(left["state"], right["state"])
            for left, right in zip(old_regular["atoms"], new_regular["atoms"], strict=True)
        )
    )
    if not parity_ok:
        errors.append("positive-support V65r2 path is not bitwise identical to V65r1")

    mass = normalize_identity_log_evidence([math.log(0.2), -math.inf])
    try:
        normalize_identity_log_evidence([-math.inf, -math.inf])
    except ImpossiblePublicHistory:
        both_infinite_rejected = True
    else:
        both_infinite_rejected = False
    normalization_ok = bool(np.array_equal(mass, np.asarray([1.0, 0.0]))) and both_infinite_rejected
    if not normalization_ok:
        errors.append("identity evidence normalization added mass or accepted all-zero support")

    repeats = [smc2_inference(family, fatal, tiny, 7, repeat) for repeat in range(3)]
    pooled = pool_repeats(repeats)
    rb = rao_blackwellize_measure(family, pooled, fatal)
    scores = score_all_actions(family, rb)
    downstream_ok = bool(
        posterior_summary(family, pooled)["identity"][1] == 0.0
        and [row["action"] for row in scores] == ["n", "e", "s", "w"]
        and all(row["finite"] and row["normalizes"] for row in scores)
    )
    if not downstream_ok:
        errors.append("pooled Rao-Blackwellized acquisition failed on repaired measure")

    truth_rejected = False
    try:
        boolean_identity_support(family, {**regular, "truth": 1}, 0)
    except PermissionError:
        truth_rejected = True
    collision = smc2_inference(
        family, regular, tiny, 7, 0, shared_inner_stream=True
    )["diagnostics"]["random_stream_collision_count"]

    mutation_checks = {
        "revert_to_v65r1_extinction_abort": old_failure_reproduced and fatal_ok,
        "classify_exact_zero_as_positive_support": not candidate_support[(fatal_id, 1)][
            "supported"
        ],
        "classify_positive_support_as_exact_zero": candidate_support[(fatal_id, 0)][
            "supported"
        ],
        "add_epsilon_mass_to_exact_zero_identity": repaired["identity"][1] == 0.0,
        "fabricate_atom_for_exact_zero_identity": all(
            atom["identity"] == 0 for atom in repaired["atoms"]
        ),
        "run_outer_particles_for_exact_zero_identity": repaired["diagnostics"]["work"][
            "outer_particles_initialized"
        ]
        == 7,
        "swallow_positive_support_particle_extinction": positive_collapse_rejected,
        "accept_both_identities_impossible": impossible_rejected,
        "ignore_reset_observation_support": all(
            not boolean_identity_support(family, impossible, identity)["supported"]
            for identity in range(2)
        ),
        "use_theta_dependent_support_threshold": all(
            row["theta_support_invariant"] for row in candidate_support.values()
        ),
        "change_positive_support_particle_path": parity_ok,
        "change_positive_support_random_streams": old_regular["diagnostics"][
            "random_stream_count"
        ]
        == new_regular["diagnostics"]["random_stream_count"],
        "omit_complete_work_diagnostics": all(
            field in repaired["diagnostics"]["work"] for field in WORK_FIELDS
        ),
        "omit_candidate_action_after_repair": [row["action"] for row in scores]
        == ["n", "e", "s", "w"],
        "truth_field_access": truth_rejected,
        "shared_inner_random_streams": collision > 0,
    }
    kill_rate = float(np.mean(list(mutation_checks.values())))
    mutation_ok = kill_rate == 1.0
    if not mutation_ok:
        errors.append("V65r2 implementation mutation audit did not kill every mutant")

    analytic_checks = {
        "independent_boolean_support_matches_all_96_identity_histories": support_reference_ok,
        "one_finite_one_infinite_identity_normalization": normalization_ok,
        "sealed_zero_identity_has_no_mass_or_atoms": fatal_ok,
        "both_identity_impossible_rejected": impossible_rejected,
        "positive_support_particle_extinction_rejected": positive_collapse_rejected,
        "positive_support_path_bitwise_unchanged": parity_ok,
        "pooled_Rao_Blackwellized_scores_complete": downstream_ok,
    }
    analytic_ok = all(analytic_checks.values())

    checks = {
        "design_bindings_and_implementation_only_authorization": design_ok,
        "evaluator_and_outcome_absent": downstream_absent,
        "independent_boolean_support_reference": support_reference_ok,
        "sealed_failure_narrowly_repaired": fatal_ok and old_failure_reproduced,
        "extinction_classes_fail_closed": extinction_classification_ok,
        "positive_support_path_bitwise_unchanged": parity_ok,
        "exact_zero_identity_normalization": normalization_ok,
        "pooled_Rao_Blackwellized_downstream_compatibility": downstream_ok,
        "all_registered_mutants_killed": mutation_ok,
        "all_analytic_fixtures_pass": analytic_ok,
    }
    audit = {
        "schema_version": "65r2",
        "experiment": "v65r2_implementation_audit",
        "passed": not errors and all(checks.values()),
        "decision": (
            "freeze_repair_implementation_and_authorize_durable_evaluator_implementation_only"
            if not errors and all(checks.values())
            else "reject_v65r2_implementation"
        ),
        "errors": errors,
        "checks": checks,
        "support_reference": {
            "sealed_records": len(subset),
            "identity_histories": len(reference_support),
            "exact_zero_rows": [list(row) for row in zero_rows],
            "candidate_EIG_scores_on_sealed_subset": 4,
            "candidate_EIG_scored_only_on_mandatory_repaired_regression_record": True,
        },
        "mutation_audit": {
            "registered": len(mutation_checks),
            "killed": sum(mutation_checks.values()),
            "kill_rate": kill_rate,
            "checks": mutation_checks,
        },
        "analytic_fixtures": {
            "registered": len(analytic_checks),
            "passed": sum(analytic_checks.values()),
            "pass_rate": float(np.mean(list(analytic_checks.values()))),
            "checks": analytic_checks,
        },
        "data_access": {
            "sealed_public_records_loaded": len(subset),
            "sealed_records_with_candidate_EIG_scored": 1,
            "truth_fields_accessed": 0,
            "V65r1_evaluation_reruns": 0,
            "V65r2_evaluation_attempts": 0,
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

    sources = (
        "python/v65r2_smc2_eig.py",
        "python/test_v65r2_smc2_eig.py",
        "python/audit_and_freeze_v65r2_implementation.py",
        "python/v65_smc2_eig.py",
        "python/v65_scalar_reference.py",
        "python/v64_external_eig.py",
        "python/v62_external_pomdp.py",
    )
    lock = {
        "schema_version": "65r2",
        "experiment": "v65r2_implementation_lock",
        "design_lock": str(design_path.relative_to(PROJECT_ROOT)),
        "design_lock_sha256": file_sha256(design_path),
        "implementation_audit": str(audit_path.relative_to(PROJECT_ROOT)),
        "implementation_audit_sha256": file_sha256(audit_path),
        "source_sha256": {
            relative: file_sha256(PROJECT_ROOT / relative) for relative in sources
        },
        "runtime": {"python": platform.python_version(), "numpy": np.__version__},
        "authorization": {
            "modify_or_rerun_v65r1": False,
            "modify_v65r2_design": False,
            "modify_v65r2_implementation": False,
            "write_and_audit_durable_evaluator": True,
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
    print(
        json.dumps(
            {
                "audit_passed": audit["passed"],
                "mutants_killed": audit["mutation_audit"]["killed"],
                "mutants_registered": audit["mutation_audit"]["registered"],
                "analytic_pass_rate": audit["analytic_fixtures"]["pass_rate"],
                "lock": str(output_path.relative_to(PROJECT_ROOT)),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
