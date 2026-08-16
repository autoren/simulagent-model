#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json

from evaluate_v54_eig import qualification
from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT


EVALUATION_FILES = (
    "python/evaluate_v54_eig.py",
    "python/audit_v54_populations.py",
    "python/seal_v54_populations.py",
    "python/audit_and_summarize_v54.py",
    "python/freeze_v54_outcome.py",
    "python/audit_v54_evaluation_implementation.py",
    "scripts/run-v54-exact-one-step-eig.sh",
)


def passing_fixture(config):
    gates = config["gates"]
    metrics = {
        "selection": {
            "completed_fraction": 1.0,
            "candidate_and_predictive_normalization_rate": 1.0,
            "finite_value_rate": 1.0,
            "maximum_absolute_candidate_eig_error": 0.0,
            "mean_absolute_candidate_eig_error": 0.0,
            "optimal_set_membership_rate": 1.0,
            "maximum_selected_eig_regret": 0.0,
            "maximum_entropy_reduction_identity_error": 0.0,
            "minimum_candidate_eig": 0.0,
            "maximum_eig_minus_prior_entropy": 0.0,
            "informative_record_fraction": gates["minimumInformativeRecordFraction"],
            "mean_oracle_minus_uniform_random_eig": gates["minimumMeanOracleMinusUniformRandomEigNats"],
            "mean_fraction_of_available_eig_captured": 1.0,
            "strictly_dominated_no_op_selection_rate": 0.0,
            "controls": {"detected_or_dominated": gates["minimumControlsDetectedOrDominated"]},
        },
        "adaptive_sbc": {
            "post_selection_normalization_rate": 1.0,
            "minimum_chi_square_p_value": 1.0,
            "maximum_absolute_rank_bin_z": 0.0,
            "maximum_absolute_coverage_z": 0.0,
            "sealed_selection_match_rate": 1.0,
        },
        "selection_integrity": {
            "truth_field_access_count": 0,
            "realized_outcome_access_before_selection_count": 0,
            "candidate_omission_count": 0,
            "canonical_tie_break_violation_count": 0,
            "history_and_outcome_stream_collision_count": 0,
        },
    }
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--implementation-lock", default="configs/v54-implementation-lock.json")
    parser.add_argument(
        "--output",
        default="outputs/v54-exact-one-step-eig/evaluation-implementation-audit.json",
    )
    args = parser.parse_args()
    implementation_path = (PROJECT_ROOT / args.implementation_lock).resolve()
    output = (PROJECT_ROOT / args.output).resolve()
    implementation = json.loads(implementation_path.read_text())
    design = json.loads((PROJECT_ROOT / implementation["design_lock"]).read_text())
    config = design["config_payload"]
    errors = []

    implementation_bound = (
        implementation["authorization"]["construct_v54_active_populations"]
        and not implementation["authorization"]["run_v54_active_evaluation"]
        and all(
            file_sha256(PROJECT_ROOT / path) == digest
            for section in ("implementation_files_sha256", "base_dependencies_sha256")
            for path, digest in implementation[section].items()
        )
    )
    if not implementation_bound:
        errors.append("V54 core implementation lock is not intact")

    fixture = passing_fixture(config)
    qualification_fixture = qualification(fixture, config["gates"])
    qualification_ok = qualification_fixture["passed"] and len(
        qualification_fixture["checks"]
    ) == 25
    if not qualification_ok:
        errors.append("V54 qualification function omits gates or rejects the boundary fixture")

    single_run_firewall = all(
        token in (PROJECT_ROOT / "python/evaluate_v54_eig.py").read_text()
        for token in (
            "evaluation already attempted", "evaluation-attempt.json",
            '"evaluation_run": 1',
        )
    )
    if not single_run_firewall:
        errors.append("V54 evaluator lacks the single-run attempt firewall")

    downstream_absent = not any(
        (PROJECT_ROOT / path).exists()
        for path in (
            "configs/v54-evaluation-implementation-lock.json",
            "configs/v54-population-seal.json",
            "configs/v54-outcome-lock.json",
            "data/v54-exact-one-step-eig",
            "outputs/v54-exact-one-step-eig/population-audit.json",
            "outputs/v54-exact-one-step-eig/evaluation-attempt.json",
            "outputs/v54-exact-one-step-eig/evaluation",
        )
    )
    if not downstream_absent:
        errors.append("V54 population or evaluation exists before evaluation implementation lock")

    audit = {
        "schema_version": 54,
        "experiment": "v54_evaluation_implementation_audit",
        "passed": not errors,
        "decision": (
            "authorize_v54_evaluation_implementation_lock" if not errors
            else "repair_v54_evaluation_implementation"
        ),
        "errors": errors,
        "implementation_lock": str(implementation_path.relative_to(PROJECT_ROOT)),
        "implementation_lock_sha256": file_sha256(implementation_path),
        "implementation_files_sha256": {
            path: file_sha256(PROJECT_ROOT / path) for path in EVALUATION_FILES
        },
        "base_dependencies_sha256": {
            "configs/v54-design-lock.json": file_sha256(PROJECT_ROOT / "configs/v54-design-lock.json"),
            "configs/v54-implementation-lock.json": file_sha256(implementation_path),
        },
        "checks": {
            "core_implementation_lock_intact": implementation_bound,
            "qualification_boundary_fixture": qualification_ok,
            "qualification_check_count": len(qualification_fixture["checks"]),
            "single_run_firewall": single_run_firewall,
            "downstream_absent": downstream_absent,
        },
        "data_access": {
            "v54_candidate_population_records_accessed": 0,
            "v54_population_generator_executions": 0,
            "v54_active_evaluation_runs": 0,
            "v54_adaptive_sbc_runs": 0,
            "model_forward_passes": 0,
            "adapter_training_runs": 0,
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    print(json.dumps(audit, indent=2, sort_keys=True))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
