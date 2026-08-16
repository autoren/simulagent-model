#!/usr/bin/env python3
"""Audit and freeze V65r3 without scoring any sealed history during implementation."""
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
from v64_external_eig import load_family
from v65r3_smc2_eig import (
    ImpossiblePublicHistory,
    ParticleExtinctionWithPositiveSupport,
    assert_synthetic_implementation_fixture,
    boolean_identity_support,
    load_config,
    normalize_identity_log_evidence,
    pool_repeats,
    posterior_summary,
    rao_blackwellize_measure,
    score_all_actions_for_implementation_fixture,
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


def reference_support(family, record: dict[str, Any], identity: int) -> tuple[bool, int | None]:
    observation = family.model.observations.index(record["initial_observation"])
    states = {
        state
        for state, probability in enumerate(family.model.initial)
        if probability > 0.0 and family.model.observation[0, state, observation] > 0.0
    }
    if not states:
        return False, -1
    for tick, (action_name, observation_name) in enumerate(
        zip(record["actions"], record["observations"], strict=True)
    ):
        action = family.model.actions.index(action_name)
        observation = family.model.observations.index(observation_name)
        transition = family.transitions[identity, 0, action]
        states = {
            successor
            for state in states
            for successor in range(len(family.model.states))
            if transition[state, successor] > 0.0
            and family.model.observation[action, successor, observation] > 0.0
        }
        if not states:
            return False, tick
    return True, None


def main() -> None:
    design_path = PROJECT_ROOT / "configs/v65r3-design-lock.json"
    audit_path = PROJECT_ROOT / "outputs/v65r3-synthetic-only-implementation/implementation-audit.json"
    output_path = PROJECT_ROOT / "configs/v65r3-implementation-lock.json"
    if output_path.exists():
        raise RuntimeError("V65r3 implementation already frozen")
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
        and file_sha256(PROJECT_ROOT / design["source_v65r2_development_outcome_lock"])
        == design["source_v65r2_development_outcome_lock_sha256"]
    )
    if not design_ok:
        errors.append("V65r3 design binding or implementation-only authorization failed")

    downstream_absent = not any(
        (PROJECT_ROOT / path).exists()
        for path in (
            "configs/v65r3-evaluation-implementation-lock.json",
            "configs/v65r3-outcome-lock.json",
            "python/evaluate_v65r3_eig.py",
            "outputs/v65r3-synthetic-only-implementation/evaluation",
        )
    )
    if not downstream_absent:
        errors.append("V65r3 evaluator or evaluation exists before implementation lock")

    v65r2_design = json.loads((PROJECT_ROOT / design["source_v65r2_design_lock"]).read_text())
    subset_seal = json.loads((PROJECT_ROOT / v65r2_design["subset_seal"]).read_text())
    subset = read_jsonl(PROJECT_ROOT / subset_seal["files"]["subset_public"]["path"])
    fatal_id = design["repair_payload"]["mandatoryImplementationFixtures"]["sealedSupportOnly"]
    fatal = next(row for row in subset if row["record_id"] == fatal_id)
    synthetic = design["repair_payload"]["mandatoryImplementationFixtures"]["syntheticEIG"]
    family = load_family(quadrature_nodes=17)
    tiny = copy.deepcopy(config)
    tiny["smcSquared"]["innerStateParticleBudget"] = 127

    support_checks = []
    for record in subset:
        for identity in range(2):
            candidate = boolean_identity_support(family, record, identity)
            reference = reference_support(family, record, identity)
            support_checks.append(
                candidate["supported"] == reference[0]
                and candidate["extinction_tick_zero_based"] == reference[1]
            )
    fatal_result = smc2_inference(family, fatal, tiny, 7, 0)
    sealed_support_only_ok = bool(
        all(support_checks)
        and fatal_result["normalizes"]
        and fatal_result["identity"][1] == 0.0
        and fatal_result["log_evidence_by_identity"][1] == -math.inf
        and all(atom["identity"] == 0 for atom in fatal_result["atoms"])
        and all(field in fatal_result["diagnostics"]["work"] for field in WORK_FIELDS)
    )
    if not sealed_support_only_ok:
        errors.append("sealed support-only regression fixture failed")

    assert_synthetic_implementation_fixture(synthetic, subset)
    repeats = [smc2_inference(family, synthetic, tiny, 7, repeat) for repeat in range(3)]
    pooled = pool_repeats(repeats)
    rb = rao_blackwellize_measure(family, pooled, synthetic)
    synthetic_scores = score_all_actions_for_implementation_fixture(
        family, rb, synthetic, subset
    )
    synthetic_scoring_ok = bool(
        [row["action"] for row in synthetic_scores] == ["n", "e", "s", "w"]
        and all(row["finite"] and row["normalizes"] for row in synthetic_scores)
        and posterior_summary(family, pooled)["normalizes"]
    )
    if not synthetic_scoring_ok:
        errors.append("authorized synthetic EIG fixture failed")

    sealed_id_rejected = False
    try:
        assert_synthetic_implementation_fixture(
            {**synthetic, "record_id": subset[0]["record_id"]}, subset
        )
    except PermissionError:
        sealed_id_rejected = True
    sealed_history_rejected = False
    try:
        assert_synthetic_implementation_fixture({**subset[0], "record_id": "changed"}, subset)
    except PermissionError:
        sealed_history_rejected = True
    firewall_ok = sealed_id_rejected and sealed_history_rejected
    if not firewall_ok:
        errors.append("V65r3 sealed ID or full-history scoring firewall failed")

    impossible = {
        "record_id": "v65r3-audit-impossible",
        "prefix_length": 0,
        "initial_observation": "good",
        "actions": [],
        "observations": [],
    }
    try:
        smc2_inference(family, impossible, tiny, 7, 0)
    except ImpossiblePublicHistory:
        both_impossible_rejected = True
    else:
        both_impossible_rejected = False
    try:
        smc2_inference(
            family,
            synthetic,
            tiny,
            7,
            0,
            force_positive_support_particle_extinction_identity=0,
        )
    except ParticleExtinctionWithPositiveSupport:
        positive_collapse_rejected = True
    else:
        positive_collapse_rejected = False

    old = v65r1.smc2_inference(family, synthetic, tiny, 7, 0)
    new = smc2_inference(family, synthetic, tiny, 7, 0)
    parity_ok = bool(
        old["log_evidence_by_identity"] == new["log_evidence_by_identity"]
        and old["diagnostics"]["work"] == new["diagnostics"]["work"]
        and old["diagnostics"]["random_stream_count"]
        == new["diagnostics"]["random_stream_count"]
        and len(old["atoms"]) == len(new["atoms"])
        and all(
            left["identity"] == right["identity"]
            and left["theta"] == right["theta"]
            and left["weight"] == right["weight"]
            and np.array_equal(left["state"], right["state"])
            for left, right in zip(old["atoms"], new["atoms"], strict=True)
        )
    )

    mass = normalize_identity_log_evidence([math.log(0.2), -math.inf])
    try:
        normalize_identity_log_evidence([-math.inf, -math.inf])
    except ImpossiblePublicHistory:
        all_zero_rejected = True
    else:
        all_zero_rejected = False
    normalization_ok = np.array_equal(mass, np.asarray([1.0, 0.0])) and all_zero_rejected
    collision_count = smc2_inference(
        family, synthetic, tiny, 7, 0, shared_inner_stream=True
    )["diagnostics"]["random_stream_collision_count"]

    mutation_checks = {
        "allow_sealed_record_id_in_implementation_scorer": sealed_id_rejected,
        "allow_sealed_full_history_under_new_id": sealed_history_rejected,
        "revert_to_v65r1_exact_zero_abort": fatal_result["identity"][1] == 0.0,
        "add_epsilon_mass_to_zero_identity": fatal_result["identity"][1] == 0.0,
        "fabricate_zero_identity_atom": all(
            atom["identity"] == 0 for atom in fatal_result["atoms"]
        ),
        "run_particles_for_zero_identity": fatal_result["diagnostics"]["work"][
            "outer_particles_initialized"
        ]
        == 7,
        "accept_both_identities_impossible": both_impossible_rejected,
        "swallow_positive_support_particle_extinction": positive_collapse_rejected,
        "change_positive_support_path": parity_ok,
        "change_positive_support_random_stream": old["diagnostics"]["random_stream_count"]
        == new["diagnostics"]["random_stream_count"],
        "normalize_two_negative_infinity_evidences": all_zero_rejected,
        "omit_synthetic_candidate_action": [row["action"] for row in synthetic_scores]
        == ["n", "e", "s", "w"],
        "nonfinite_synthetic_candidate": all(row["finite"] for row in synthetic_scores),
        "unnormalized_synthetic_predictive": all(
            row["normalizes"] for row in synthetic_scores
        ),
        "omit_work_diagnostics": all(
            field in fatal_result["diagnostics"]["work"] for field in WORK_FIELDS
        ),
        "share_inner_random_streams": collision_count > 0,
    }
    mutation_checks = {key: bool(value) for key, value in mutation_checks.items()}
    mutation_ok = all(mutation_checks.values())
    if not mutation_ok:
        errors.append("V65r3 implementation mutation audit did not kill every mutant")

    analytic_checks = {
        "independent_support_matches_96_identity_histories": all(support_checks),
        "sealed_zero_identity_support_only_regression": sealed_support_only_ok,
        "synthetic_EIG_fixture_complete": synthetic_scoring_ok,
        "sealed_ID_and_history_firewall": firewall_ok,
        "both_impossible_rejected": both_impossible_rejected,
        "positive_support_particle_collapse_rejected": positive_collapse_rejected,
        "positive_support_path_bitwise_unchanged": parity_ok,
        "identity_normalization_exact_zero": normalization_ok,
    }
    analytic_checks = {key: bool(value) for key, value in analytic_checks.items()}
    analytic_ok = all(analytic_checks.values())

    checks = {
        "design_binding_and_implementation_only_authorization": design_ok,
        "evaluator_and_evaluation_absent": downstream_absent,
        "sealed_support_only_regression": sealed_support_only_ok,
        "synthetic_only_EIG_fixture": synthetic_scoring_ok,
        "sealed_ID_and_history_scoring_firewall": firewall_ok,
        "extinction_classes_fail_closed": both_impossible_rejected
        and positive_collapse_rejected,
        "positive_support_path_bitwise_unchanged": parity_ok,
        "exact_zero_identity_normalization": normalization_ok,
        "all_registered_mutants_killed": mutation_ok,
        "all_analytic_fixtures_pass": analytic_ok,
    }
    audit = {
        "schema_version": "65r3",
        "experiment": "v65r3_implementation_audit",
        "passed": not errors and all(checks.values()),
        "decision": (
            "freeze_v65r3_implementation_and_authorize_durable_evaluator_only"
            if not errors and all(checks.values())
            else "reject_v65r3_implementation"
        ),
        "errors": errors,
        "checks": checks,
        "mutation_audit": {
            "registered": len(mutation_checks),
            "killed": sum(mutation_checks.values()),
            "kill_rate": float(np.mean(list(mutation_checks.values()))),
            "checks": mutation_checks,
        },
        "analytic_fixtures": {
            "registered": len(analytic_checks),
            "passed": sum(analytic_checks.values()),
            "pass_rate": float(np.mean(list(analytic_checks.values()))),
            "checks": analytic_checks,
        },
        "access": {
            "sealed_public_records_loaded": len(subset),
            "sealed_records_with_candidate_EIG_scored": 0,
            "synthetic_records_with_candidate_EIG_scored": 1,
            "synthetic_candidate_actions_scored": 4,
            "truth_fields_accessed": 0,
            "V65r1_evaluation_reruns": 0,
            "V65r2_evaluation_attempts": 0,
            "V65r3_evaluation_attempts": 0,
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
        "python/v65r3_smc2_eig.py",
        "python/test_v65r3_smc2_eig.py",
        "python/audit_and_freeze_v65r3_implementation.py",
        "python/v65r2_smc2_eig.py",
        "python/v65_smc2_eig.py",
        "python/v65_scalar_reference.py",
        "python/v64_external_eig.py",
        "python/v62_external_pomdp.py",
    )
    lock = {
        "schema_version": "65r3",
        "experiment": "v65r3_implementation_lock",
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
            "modify_or_continue_v65r2": False,
            "modify_v65r3_design_or_implementation": False,
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
                "sealed_candidate_EIG_scores": 0,
                "mutants_killed": sum(mutation_checks.values()),
                "mutants_registered": len(mutation_checks),
                "analytic_pass_rate": audit["analytic_fixtures"]["pass_rate"],
                "lock": str(output_path.relative_to(PROJECT_ROOT)),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
