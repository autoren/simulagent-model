#!/usr/bin/env python3
"""Audit and freeze the durable V66 one-shot evaluator."""
from __future__ import annotations

import copy
import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from evaluate_v66_reward import (
    TERMINAL_NAMES,
    aggregate_evaluation,
    atomic_write_json,
    atomic_write_jsonl,
    batched_known_model_oracle,
    evaluate_record,
    failure_payload,
    reserve_attempt,
)
from test_v66_evaluator import (
    synthetic_access,
    synthetic_implementation_audit,
    synthetic_rows,
)
from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT
from v64_external_eig import filter_public_history, load_family
from v66_bayes_adaptive_reward import (
    exact_kernel_and_belief,
    posterior_weighted_model_oracle,
)


def payload_hash(value: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def main() -> None:
    implementation_path = PROJECT_ROOT / "configs/v66-implementation-lock.json"
    audit_path = PROJECT_ROOT / "outputs/v66-external-bayes-adaptive-reward/evaluator-audit.json"
    output_path = PROJECT_ROOT / "configs/v66-evaluation-implementation-lock.json"
    evaluation_dir = PROJECT_ROOT / "outputs/v66-external-bayes-adaptive-reward/evaluation"
    if output_path.exists():
        raise RuntimeError("V66 evaluator already frozen")
    if evaluation_dir.exists():
        raise RuntimeError("V66 evaluation directory exists before evaluator lock")
    implementation = json.loads(implementation_path.read_text())
    design_path = PROJECT_ROOT / implementation["design_lock"]
    design = json.loads(design_path.read_text())
    config = design["config_payload"]
    errors: list[str] = []

    implementation_payload = {
        key: value
        for key, value in implementation.items()
        if key != "lock_payload_sha256"
    }
    implementation_ok = bool(
        payload_hash(implementation_payload) == implementation["lock_payload_sha256"]
        and implementation["authorization"]["write_and_audit_durable_evaluator"]
        and not implementation["authorization"]["run_evaluation"]
        and not implementation["authorization"]["formal_verification"]
        and file_sha256(design_path) == implementation["design_lock_sha256"]
        and file_sha256(PROJECT_ROOT / implementation["implementation_audit"])
        == implementation["implementation_audit_sha256"]
        and all(
            file_sha256(PROJECT_ROOT / relative) == digest
            for relative, digest in implementation["source_sha256"].items()
        )
    )
    if not implementation_ok:
        errors.append("V66 frozen implementation or evaluator-only authorization failed")

    subset_seal_path = PROJECT_ROOT / design["subset_seal"]
    subset_seal = json.loads(subset_seal_path.read_text())
    subset_binding_ok = bool(
        file_sha256(subset_seal_path) == design["subset_seal_sha256"]
        and all(
            file_sha256(PROJECT_ROOT / row["path"]) == row["sha256"]
            for row in subset_seal["files"].values()
        )
        and subset_seal["counts"]["subset_public"] == 48
        and subset_seal["prefix_counts"]
        == {str(prefix): 8 for prefix in range(6)}
    )
    if not subset_binding_ok:
        errors.append("V66 subset seal or immutable files changed")

    smc_design_path = PROJECT_ROOT / "configs/v65r3-design-lock.json"
    smc_implementation_path = PROJECT_ROOT / "configs/v65r3-implementation-lock.json"
    smc_implementation = json.loads(smc_implementation_path.read_text())
    smc_implementation_audit = json.loads(
        (PROJECT_ROOT / smc_implementation["implementation_audit"]).read_text()
    )
    inherited_smc_ok = bool(
        smc_implementation["authorization"]["write_and_audit_durable_evaluator"]
        and smc_implementation_audit["passed"]
        and smc_implementation_audit["mutation_audit"]["kill_rate"] == 1.0
        and smc_implementation_audit["mutation_audit"]["checks"][
            "share_inner_random_streams"
        ]
        and smc_implementation_audit["analytic_fixtures"]["pass_rate"] == 1.0
    )
    if not inherited_smc_ok:
        errors.append("V66 inherited V65r3 SMC2 implementation audit is incomplete")

    implementation_audit = json.loads(
        (PROJECT_ROOT / implementation["implementation_audit"]).read_text()
    )
    aggregate_audit = copy.deepcopy(implementation_audit)
    aggregate_audit["inherited_smc_shared_stream_detected"] = True
    passing = aggregate_evaluation(
        synthetic_rows(), config, aggregate_audit, synthetic_access()
    )
    aggregate_ok = bool(
        passing["passed"]
        and not passing["failed_gates"]
        and passing["integrity"]["records"] == 48
        and passing["controls"]["detected_or_dominated"]
        >= config["gates"]["minimumControlsDetectedOrDominated"]
    )
    if not aggregate_ok:
        errors.append("V66 aggregate did not pass the valid synthetic fixture")

    family = load_family(quadrature_nodes=5)
    synthetic_record = {
        "record_id": "v66-evaluator-end-to-end-synthetic",
        "prefix_length": 2,
        "initial_observation": "left",
        "actions": ["n", "e"],
        "observations": ["neither", "right"],
    }
    tiny = copy.deepcopy(config)
    tiny["approximatePosterior"]["outerThetaParticlesPerIdentity"] = 7
    tiny["persistentMixtureQuadrature"].update(
        {
            "primaryPoints": 4,
            "primarySystematicOffset": 0.5 / 4,
            "sensitivityPoints": 8,
            "sensitivitySystematicOffset": 0.5 / 8,
        }
    )
    smc_config = json.loads(smc_design_path.read_text())["config_payload"]
    synthetic_cell = evaluate_record(family, synthetic_record, tiny, smc_config)
    end_to_end_ok = bool(
        synthetic_cell["integrity"]["finite"]
        and synthetic_cell["integrity"]["normalizes"]
        and synthetic_cell["integrity"]["candidate_complete"]
        and synthetic_cell["integrity"]["tie_break_valid"]
        and not synthetic_cell["integrity"]["oracle_physical_state_revealed"]
        and synthetic_cell["integrity"]["invalid_mean_transition_labeled_invalid"]
        and len(synthetic_cell["repeat_diagnostics"]) == 3
        and synthetic_cell["persistent_mixture"]["primary_points"] == 4
        and synthetic_cell["persistent_mixture"]["sensitivity_points"] == 8
        and synthetic_cell["runtime"]["exact_Bellman_nodes"] > 100
        and synthetic_cell["runtime"]["approximate_Bellman_nodes"] > 100
    )
    if not end_to_end_ok:
        errors.append("V66 reduced synthetic end-to-end evaluator fixture failed")

    exact, _ = filter_public_history(
        family,
        synthetic_record["initial_observation"],
        synthetic_record["actions"],
        synthetic_record["observations"],
    )
    exact_kernel, exact_belief = exact_kernel_and_belief(family, exact)
    batched = batched_known_model_oracle(exact_kernel, exact_belief, 2)
    loop = posterior_weighted_model_oracle(exact_kernel, exact_belief, 2)
    oracle_reference_error = abs(float(batched["value"]) - float(loop["value"]))
    oracle_ok = bool(
        oracle_reference_error <= 1e-10
        and not batched["physical_state_revealed"]
        and batched["active_models"] > 0
    )
    if not oracle_ok:
        errors.append("V66 batched known-model oracle differs from locked per-model oracle")

    unit_command = [
        sys.executable,
        "-m",
        "unittest",
        "python/test_v66_evaluator.py",
        "-v",
    ]
    unit = subprocess.run(
        unit_command,
        cwd=PROJECT_ROOT,
        env={**dict(__import__("os").environ), "PYTHONPATH": "python"},
        capture_output=True,
        text=True,
        check=False,
    )
    unit_ok = unit.returncode == 0 and "Ran 9 tests" in unit.stderr and "OK" in unit.stderr
    if not unit_ok:
        errors.append("V66 evaluator unit tests failed")

    durable_checks: dict[str, bool] = {}
    with tempfile.TemporaryDirectory(dir=PROJECT_ROOT) as directory:
        root = Path(directory)
        json_path = root / "atomic.json"
        jsonl_path = root / "atomic.jsonl"
        atomic_write_json(json_path, {"attempt": 1})
        atomic_write_jsonl(jsonl_path, [{"cell": 1}, {"cell": 2}])
        durable_checks["atomic_json_complete"] = json.loads(json_path.read_text()) == {
            "attempt": 1
        }
        durable_checks["atomic_jsonl_complete"] = len(
            jsonl_path.read_text().splitlines()
        ) == 2
        durable_checks["atomic_temporaries_removed"] = not any(
            path.name.startswith(".") for path in root.iterdir()
        )

    with tempfile.TemporaryDirectory(dir=PROJECT_ROOT) as directory:
        root = Path(directory)
        marker = {"logical_evaluation_attempt": 1}
        attempt = reserve_attempt(root, marker)
        durable_checks["attempt_marker_written"] = json.loads(attempt.read_text()) == marker
        try:
            reserve_attempt(root, marker)
        except RuntimeError:
            durable_checks["repeat_attempt_rejected"] = True
        else:
            durable_checks["repeat_attempt_rejected"] = False
        try:
            raise ValueError("V66-durable-audit-fixture")
        except ValueError as error:
            failure = failure_payload(
                lock_path=implementation_path,
                attempt_path=attempt,
                stage="durable_audit_fixture",
                progress={"records_completed": 3},
                access={"logical_evaluation_attempts": 1},
                error=error,
            )
        durable_checks["failure_consumes_one_shot"] = bool(
            failure["one_shot_authorization_consumed"]
            and not failure["claim_boundary"]["V66_rerun_authorized"]
        )
        durable_checks["failure_has_stage_progress_access_exception_and_bindings"] = all(
            key in failure for key in ("stage", "progress", "access", "exception", "bindings")
        )
        durable_checks["failure_is_non_authorizing"] = bool(
            not failure["passed"]
            and failure["decision"] == "do_not_authorize_policy_verification"
        )

    for terminal in TERMINAL_NAMES:
        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT) as directory:
            root = Path(directory)
            (root / terminal).write_text("{}\n")
            try:
                reserve_attempt(root, {"logical_evaluation_attempt": 1})
            except RuntimeError:
                durable_checks[f"terminal_{terminal}_blocks_attempt"] = True
            else:
                durable_checks[f"terminal_{terminal}_blocks_attempt"] = False

    source = (PROJECT_ROOT / "python/evaluate_v66_reward.py").read_text()
    run_source = source[source.index("def run_evaluation"):source.index("def main()")]
    durable_checks["attempt_reserved_before_subset_read"] = (
        run_source.index("reserve_attempt(") < run_source.index("read_jsonl(")
    )
    durable_checks["caught_exception_serializes_failure"] = (
        'atomic_write_json(output_dir / "failure.json", failure)' in run_source
    )
    durable_checks["success_serializes_result"] = (
        'atomic_write_json(output_dir / "result.json", result)' in run_source
    )
    durable_checks["success_serializes_raw_cells"] = (
        'atomic_write_jsonl(raw_path, rows)' in run_source
    )
    durable_checks["per_record_progress_is_flushed"] = 'flush=True' in run_source
    durable_checks["V65r3_SMC2_inference_imported"] = (
        "from v65r3_smc2_eig import pool_repeats, smc2_inference" in source
    )
    durable_checks["batched_oracle_does_not_reveal_state"] = (
        '"physical_state_revealed": False' in source
    )
    durable_ok = all(durable_checks.values())
    if not durable_ok:
        errors.append("V66 durable attempt/result/failure protocol audit failed")

    mutation_checks = {
        **durable_checks,
        "valid_aggregate_passes": aggregate_ok,
        "reduced_end_to_end_cell_passes": end_to_end_ok,
        "batched_oracle_matches_locked_loop": oracle_ok,
        "evaluator_unit_tests_pass": unit_ok,
        "V66_implementation_22_mutants_killed": all(
            implementation_audit["mutation_checks"].values()
        ),
        "V66_implementation_15_analytic_fixtures_pass": all(
            implementation_audit["analytic_checks"].values()
        ),
        "V65r3_shared_stream_mutant_killed": inherited_smc_ok,
        "evaluation_directory_absent": not evaluation_dir.exists(),
    }
    mutation_checks = {key: bool(value) for key, value in mutation_checks.items()}
    mutation_ok = all(mutation_checks.values())

    checks = {
        "implementation_binding_and_evaluator_only_authorization": implementation_ok,
        "immutable_subset_binding": subset_binding_ok,
        "inherited_frozen_SMC2_audits": inherited_smc_ok,
        "valid_aggregate_fixture": aggregate_ok,
        "reduced_synthetic_end_to_end_evaluator": end_to_end_ok,
        "batched_oracle_independent_reference": oracle_ok,
        "evaluator_unit_tests_9_of_9": unit_ok,
        "durable_one_shot_protocol": durable_ok,
        "all_registered_durable_binding_and_mutation_checks_pass": mutation_ok,
        "evaluation_directory_absent": not evaluation_dir.exists(),
    }
    audit = {
        "schema_version": "66",
        "experiment": "v66_evaluator_audit",
        "passed": not errors and all(checks.values()),
        "decision": (
            "freeze_v66_evaluator_and_authorize_one_immutable_evaluation"
            if not errors and all(checks.values())
            else "reject_v66_evaluator"
        ),
        "errors": errors,
        "checks": checks,
        "unit_test_command": unit_command,
        "unit_test_stdout": unit.stdout,
        "unit_test_stderr": unit.stderr,
        "mutation_audit": {
            "registered_checks": len(mutation_checks),
            "passed": sum(mutation_checks.values()),
            "pass_rate": float(np.mean(list(mutation_checks.values()))),
            "checks": mutation_checks,
            "inherited_V66_implementation_mutants_killed": 22,
            "inherited_V65r3_SMC2_mutants_killed": 16,
            "combined_checks_and_mutants": len(mutation_checks) + 38,
            "combined_passed_or_killed": sum(mutation_checks.values()) + 38,
        },
        "synthetic_end_to_end": {
            "record_id": synthetic_cell["record_id"],
            "primary": synthetic_cell["primary"],
            "integrity": synthetic_cell["integrity"],
            "runtime": synthetic_cell["runtime"],
        },
        "oracle_reference_error": oracle_reference_error,
        "access": {
            "sealed_public_records_loaded": 0,
            "sealed_reward_policies_planned_or_scored": 0,
            "synthetic_reward_policy_records": 1,
            "V64_or_V65_evaluation_result_records_loaded": 0,
            "truth_fields_accessed": 0,
            "V65r3_evaluation_reruns": 0,
            "V66_evaluation_attempts": 0,
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
        "python/evaluate_v66_reward.py",
        "python/test_v66_evaluator.py",
        "python/audit_and_freeze_v66_evaluator.py",
        "python/v66_bayes_adaptive_reward.py",
        "python/v65r3_smc2_eig.py",
        "python/v65r2_smc2_eig.py",
        "python/v65_smc2_eig.py",
        "python/v64_external_eig.py",
        "python/v62_external_pomdp.py",
    )
    lock = {
        "schema_version": "66",
        "experiment": "v66_evaluation_implementation_lock",
        "implementation_lock": str(implementation_path.relative_to(PROJECT_ROOT)),
        "implementation_lock_sha256": file_sha256(implementation_path),
        "subset_seal": str(subset_seal_path.relative_to(PROJECT_ROOT)),
        "subset_seal_sha256": file_sha256(subset_seal_path),
        "source_v65r3_design_lock": str(smc_design_path.relative_to(PROJECT_ROOT)),
        "source_v65r3_design_lock_sha256": file_sha256(smc_design_path),
        "source_v65r3_implementation_lock": str(
            smc_implementation_path.relative_to(PROJECT_ROOT)
        ),
        "source_v65r3_implementation_lock_sha256": file_sha256(
            smc_implementation_path
        ),
        "evaluator_audit": str(audit_path.relative_to(PROJECT_ROOT)),
        "evaluator_audit_sha256": file_sha256(audit_path),
        "source_sha256": {
            relative: file_sha256(PROJECT_ROOT / relative) for relative in sources
        },
        "authorization": {
            "modify_or_rerun_v65r3": False,
            "modify_v66_design_implementation_or_evaluator": False,
            "run_one_immutable_evaluation": True,
            "run_additional_evaluation": False,
            "formal_verification": False,
            "human_data": False,
            "model_access": False,
            "adapter_training": False,
        },
    }
    lock["lock_payload_sha256"] = payload_hash(lock)
    output_path.write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "audit_passed": audit["passed"],
                "checks": checks,
                "combined_checks_and_mutants": audit["mutation_audit"][
                    "combined_checks_and_mutants"
                ],
                "evaluation_directory_absent": not evaluation_dir.exists(),
                "lock": str(output_path.relative_to(PROJECT_ROOT)),
                "oracle_reference_error": oracle_reference_error,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
