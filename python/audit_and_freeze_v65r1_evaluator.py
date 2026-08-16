#!/usr/bin/env python3
"""Audit and freeze the one-shot V65r1 evaluation implementation."""
from __future__ import annotations

import copy
import hashlib
import json
import platform
from pathlib import Path
from typing import Callable

import numpy as np

from evaluate_v65r1_eig import (
    WORK_FIELDS,
    _selection_from_values,
    aggregate_evaluation,
    q95,
    weighted_wasserstein_1,
)
from test_v65r1_evaluator import (
    synthetic_access,
    synthetic_implementation_audit,
    synthetic_rows,
)
from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT


Mutation = Callable[[list[dict], dict, dict], None]


def main() -> None:
    subset_seal_path = PROJECT_ROOT / "configs/v65r1-subset-seal.json"
    audit_path = (
        PROJECT_ROOT
        / "outputs/v65r1-nested-predictive-repair/evaluation-implementation-audit.json"
    )
    output_path = PROJECT_ROOT / "configs/v65r1-evaluation-implementation-lock.json"
    evaluation_path = (
        PROJECT_ROOT / "outputs/v65r1-nested-predictive-repair/evaluation/result.json"
    )
    if output_path.exists():
        raise RuntimeError("V65r1 evaluation implementation already frozen")
    if evaluation_path.exists() or evaluation_path.parent.exists():
        raise RuntimeError("V65r1 evaluation artifacts exist before evaluator freeze")

    subset_seal = json.loads(subset_seal_path.read_text())
    implementation_path = PROJECT_ROOT / subset_seal["implementation_lock"]
    implementation = json.loads(implementation_path.read_text())
    design_path = PROJECT_ROOT / implementation["design_lock"]
    design = json.loads(design_path.read_text())
    config = design["config_payload"]
    errors: list[str] = []

    upstream_ok = bool(
        file_sha256(implementation_path) == subset_seal["implementation_lock_sha256"]
        and hashlib.sha256(
            json.dumps(
                {
                    key: value
                    for key, value in subset_seal.items()
                    if key != "lock_payload_sha256"
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest()
        == subset_seal["lock_payload_sha256"]
        and subset_seal["authorization"]["write_and_audit_evaluator"]
        and not subset_seal["authorization"]["run_evaluation"]
        and not subset_seal["authorization"]["reward_planning"]
        and file_sha256(PROJECT_ROOT / subset_seal["subset_audit"])
        == subset_seal["subset_audit_sha256"]
        and all(
            file_sha256(PROJECT_ROOT / row["path"]) == row["sha256"]
            for row in subset_seal["files"].values()
        )
        and all(
            file_sha256(PROJECT_ROOT / relative) == digest
            for relative, digest in implementation["source_sha256"].items()
        )
    )
    if not upstream_ok:
        errors.append("sealed subset or frozen SMC2 implementation binding failed")

    rows = synthetic_rows()
    implementation_audit = synthetic_implementation_audit()
    access = synthetic_access()
    passing = aggregate_evaluation(rows, config, implementation_audit, access)
    passing_fixture_ok = bool(
        passing["passed"]
        and not passing["failed_gates"]
        and len(passing["compute_diagnostics"]["cells"]) == 432
        and passing["controls"]["detected_or_dominated"] >= 6
    )
    if not passing_fixture_ok:
        errors.append("valid synthetic evaluator fixture did not pass")

    def set_primary(field: str, value) -> Mutation:
        def mutation(candidate_rows, _candidate_audit, _candidate_access) -> None:
            for row in candidate_rows:
                if row["budget"] == 509:
                    row[field] = copy.deepcopy(value)

        return mutation

    def incomplete(candidate_rows, _audit, _access) -> None:
        candidate_rows.pop()

    def duplicate(candidate_rows, _audit, _access) -> None:
        candidate_rows[-1] = copy.deepcopy(candidate_rows[0])

    def missing_repeat(candidate_rows, _audit, _access) -> None:
        candidate_rows[0]["repeat_diagnostics"].pop()

    def disable_controls(candidate_rows, _audit, _access) -> None:
        for row in candidate_rows:
            for control in row["controls"].values():
                control["exact_regret"] = 0.0
                control["strict_optimal_membership"] = True
                control["mean_absolute_eig_error"] = row["mean_absolute_eig_error"]

    def invert_scaling(candidate_rows, _audit, _access) -> None:
        for row in candidate_rows:
            value = 0.0001 if row["budget"] == 31 else 0.002
            row["absolute_eig_errors"] = [value] * 4

    def weaken_mutation_audit(_rows, audit, _access) -> None:
        audit["mutation_audit"]["kill_rate"] = 0.99

    def weaken_analytic_audit(_rows, audit, _access) -> None:
        audit["analytic_fixtures"]["pass_rate"] = 0.99

    def missing_work(candidate_rows, _audit, _access) -> None:
        del candidate_rows[0]["repeat_diagnostics"][0]["work"][WORK_FIELDS[0]]

    mutations: dict[str, Mutation] = {
        "incomplete_record_budget_grid": incomplete,
        "duplicate_record_budget_cell": duplicate,
        "missing_repeat_cell": missing_repeat,
        "pooled_posterior_not_normalized": set_primary("pooled_normalizes", False),
        "candidate_predictive_not_normalized": set_primary(
            "candidate_predictive_normalizes", False
        ),
        "nonfinite_candidate_values": set_primary("finite", False),
        "identity_TV_above_gate": set_primary("identity_tv", 0.20),
        "theta_W1_above_gate": set_primary("theta_wasserstein", 0.20),
        "joint_identity_theta_TV_above_gate": set_primary(
            "joint_identity_theta_tv", 0.40
        ),
        "state_TV_above_gate": set_primary("state_tv", 0.30),
        "candidate_predictive_TV_above_gate": set_primary(
            "candidate_predictive_tvs", [0.25] * 4
        ),
        "EIG_vector_error_above_gate": set_primary(
            "absolute_eig_errors", [0.03] * 4
        ),
        "strict_membership_failure": set_primary("strict_optimal_membership", False),
        "epsilon_membership_failure": set_primary("epsilon_optimal_membership", False),
        "selection_regret_above_gate": set_primary("exact_regret", 0.03),
        "budget_scaling_inverted": invert_scaling,
        "controls_not_detected": disable_controls,
        "compute_work_counter_missing": missing_work,
        "upstream_mutant_kill_rate_weakened": weaken_mutation_audit,
        "upstream_analytic_pass_rate_weakened": weaken_analytic_audit,
    }
    for field in (
        "logical_evaluation_attempts",
        "v64_source_public_records_loaded_during_evaluation",
        "v64_selection_audit_records_loaded",
        "v64_evaluation_records_loaded",
        "truth_field_access_count",
        "realized_outcome_access_before_selection_count",
        "candidate_omission_count",
        "tie_break_violation_count",
        "random_stream_collision_count",
        "human_record_access_count",
        "model_forward_pass_count",
        "adapter_training_run_count",
    ):
        def access_mutation(_rows, _audit, candidate_access, field=field) -> None:
            candidate_access[field] = 2 if field == "logical_evaluation_attempts" else 1

        mutations[f"access_{field}"] = access_mutation

    mutation_checks = {}
    for name, mutate in mutations.items():
        candidate_rows = copy.deepcopy(rows)
        candidate_audit = copy.deepcopy(implementation_audit)
        candidate_access = copy.deepcopy(access)
        try:
            mutate(candidate_rows, candidate_audit, candidate_access)
            result = aggregate_evaluation(
                candidate_rows, config, candidate_audit, candidate_access
            )
            mutation_checks[name] = not result["passed"]
        except (KeyError, IndexError, ValueError, ZeroDivisionError):
            mutation_checks[name] = True
    mutation_kill_rate = float(np.mean(list(mutation_checks.values())))
    mutation_audit_ok = mutation_kill_rate == 1.0
    if not mutation_audit_ok:
        errors.append("evaluator mutation audit did not kill every registered mutant")

    analytic_checks = {
        "weighted_W1_point_masses": abs(
            weighted_wasserstein_1([0.0], [1.0], [1.0], [1.0]) - 1.0
        )
        <= 1e-15,
        "weighted_W1_identical_distribution": abs(
            weighted_wasserstein_1(
                [0.0, 1.0], [0.5, 0.5], [0.0, 1.0], [0.5, 0.5]
            )
        )
        <= 1e-15,
        "linear_q95": abs(q95(list(range(20))) - 18.05) <= 1e-12,
        "canonical_tie_break": _selection_from_values(
            ["n", "e", "s", "w"], [1.0, 1.0, 0.0, 0.0]
        )
        == "n",
        "synthetic_compute_cells_complete": len(
            passing["compute_diagnostics"]["cells"]
        )
        == 432,
        "synthetic_controls_detected": passing["controls"]["detected_or_dominated"]
        == 9,
    }
    analytic_ok = all(analytic_checks.values())
    if not analytic_ok:
        errors.append("evaluator analytic fixture failed")

    checks = {
        "sealed_subset_and_implementation_bindings": upstream_ok,
        "evaluation_artifacts_absent": not evaluation_path.parent.exists(),
        "valid_synthetic_fixture_passes": passing_fixture_ok,
        "all_evaluator_mutants_killed": mutation_audit_ok,
        "all_analytic_fixtures_pass": analytic_ok,
    }
    audit = {
        "schema_version": "65r1",
        "experiment": "v65r1_evaluation_implementation_audit",
        "passed": not errors and all(checks.values()),
        "decision": (
            "freeze_evaluator_and_authorize_one_immutable_evaluation"
            if not errors and all(checks.values())
            else "reject_v65r1_evaluator"
        ),
        "errors": errors,
        "checks": checks,
        "passing_fixture": {
            "passed": passing["passed"],
            "failed_gates": passing["failed_gates"],
            "record_budget_rows": len(rows),
            "record_budget_repeat_cells": len(passing["compute_diagnostics"]["cells"]),
            "controls_detected_or_dominated": passing["controls"][
                "detected_or_dominated"
            ],
        },
        "mutation_audit": {
            "registered": len(mutation_checks),
            "killed": sum(mutation_checks.values()),
            "kill_rate": mutation_kill_rate,
            "checks": mutation_checks,
        },
        "analytic_fixtures": {
            "registered": len(analytic_checks),
            "passed": sum(analytic_checks.values()),
            "pass_rate": float(np.mean(list(analytic_checks.values()))),
            "checks": analytic_checks,
        },
        "data_access": {
            "subset_records_loaded": 0,
            "V64_records_loaded": 0,
            "truth_fields_accessed": 0,
            "human_records": 0,
            "model_forward_passes": 0,
            "adapter_training_runs": 0,
            "evaluation_attempts": 0,
        },
    }
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    if not audit["passed"]:
        print(json.dumps(audit, indent=2, sort_keys=True))
        raise SystemExit(1)

    sources = (
        "python/evaluate_v65r1_eig.py",
        "python/test_v65r1_evaluator.py",
        "python/audit_and_freeze_v65r1_evaluator.py",
        "python/v65_smc2_eig.py",
        "python/v64_external_eig.py",
        "python/v62_external_pomdp.py",
    )
    lock_payload = {
        "schema_version": "65r1",
        "experiment": "v65r1_evaluation_implementation_lock",
        "subset_seal": str(subset_seal_path.relative_to(PROJECT_ROOT)),
        "subset_seal_sha256": file_sha256(subset_seal_path),
        "implementation_lock": str(implementation_path.relative_to(PROJECT_ROOT)),
        "implementation_lock_sha256": file_sha256(implementation_path),
        "evaluation_implementation_audit": str(audit_path.relative_to(PROJECT_ROOT)),
        "evaluation_implementation_audit_sha256": file_sha256(audit_path),
        "source_sha256": {
            relative: file_sha256(PROJECT_ROOT / relative) for relative in sources
        },
        "runtime": {"python": platform.python_version(), "numpy": np.__version__},
        "authorization": {
            "modify_subset": False,
            "modify_implementation": False,
            "modify_evaluator": False,
            "run_one_immutable_evaluation": True,
            "run_additional_evaluation": False,
            "reward_planning": False,
            "formal_verification": False,
            "human_data": False,
            "model_access": False,
            "adapter_training": False,
        },
    }
    lock_payload["lock_payload_sha256"] = hashlib.sha256(
        json.dumps(lock_payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    output_path.write_text(json.dumps(lock_payload, indent=2, sort_keys=True) + "\n")
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
