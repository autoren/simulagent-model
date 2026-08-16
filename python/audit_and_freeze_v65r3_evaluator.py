#!/usr/bin/env python3
"""Audit and freeze the durable V65r3 one-shot evaluator."""
from __future__ import annotations

import copy
import hashlib
import json
import tempfile
from pathlib import Path

import numpy as np

from evaluate_v65r1_eig import aggregate_evaluation
from evaluate_v65r3_eig import (
    TERMINAL_NAMES,
    aggregate_implementation_audit,
    atomic_write_json,
    atomic_write_jsonl,
    failure_payload,
    reserve_attempt,
)
from test_v65r1_evaluator import synthetic_access, synthetic_rows
from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT


def main() -> None:
    implementation_path = PROJECT_ROOT / "configs/v65r3-implementation-lock.json"
    audit_path = PROJECT_ROOT / "outputs/v65r3-synthetic-only-implementation/evaluator-audit.json"
    output_path = PROJECT_ROOT / "configs/v65r3-evaluation-implementation-lock.json"
    evaluation_dir = PROJECT_ROOT / "outputs/v65r3-synthetic-only-implementation/evaluation"
    if output_path.exists():
        raise RuntimeError("V65r3 evaluator already frozen")
    if evaluation_dir.exists():
        raise RuntimeError("V65r3 evaluation directory exists before evaluator lock")
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
        hashlib.sha256(
            json.dumps(
                implementation_payload, sort_keys=True, separators=(",", ":")
            ).encode()
        ).hexdigest()
        == implementation["lock_payload_sha256"]
        and implementation["authorization"]["write_and_audit_durable_evaluator"]
        and not implementation["authorization"]["run_evaluation"]
        and not implementation["authorization"]["reward_planning"]
        and file_sha256(design_path) == implementation["design_lock_sha256"]
        and file_sha256(PROJECT_ROOT / implementation["implementation_audit"])
        == implementation["implementation_audit_sha256"]
        and all(
            file_sha256(PROJECT_ROOT / relative) == digest
            for relative, digest in implementation["source_sha256"].items()
        )
    )
    if not implementation_ok:
        errors.append("V65r3 frozen implementation or evaluator-only authorization failed")

    v65r2_design = json.loads((PROJECT_ROOT / design["source_v65r2_design_lock"]).read_text())
    subset_seal_path = PROJECT_ROOT / v65r2_design["subset_seal"]
    subset_seal = json.loads(subset_seal_path.read_text())
    subset_binding_ok = bool(
        file_sha256(subset_seal_path) == v65r2_design["subset_seal_sha256"]
        and all(
            file_sha256(PROJECT_ROOT / row["path"]) == row["sha256"]
            for row in subset_seal["files"].values()
        )
        and subset_seal["counts"]["subset_public"] == 48
        and subset_seal["prefix_counts"]
        == {str(prefix): 8 for prefix in range(6)}
    )
    if not subset_binding_ok:
        errors.append("V65r3 subset seal or immutable files changed")

    implementation_audit_raw = json.loads(
        (PROJECT_ROOT / implementation["implementation_audit"]).read_text()
    )
    implementation_audit = aggregate_implementation_audit(implementation_audit_raw)
    rows = synthetic_rows()
    passing = aggregate_evaluation(
        rows, config, implementation_audit, synthetic_access()
    )
    aggregate_ok = bool(
        passing["passed"]
        and not passing["failed_gates"]
        and len(passing["compute_diagnostics"]["cells"]) == 432
        and implementation_audit["mutation_audit"]["checks"][
            "share_inner_streams_across_outer_particles"
        ]
    )
    if not aggregate_ok:
        errors.append("V65r3 inherited aggregate did not pass the valid synthetic fixture")

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
            raise ValueError("durable-audit-fixture")
        except ValueError as error:
            failure = failure_payload(
                lock_path=implementation_path,
                attempt_path=attempt,
                stage="durable_audit_fixture",
                progress={"record_budget_rows_completed": 3},
                access={"logical_evaluation_attempts": 1},
                error=error,
            )
        durable_checks["failure_consumes_one_shot"] = bool(
            failure["one_shot_authorization_consumed"]
            and not failure["claim_boundary"]["V65r3_rerun_authorized"]
        )
        durable_checks["failure_has_stage_progress_access_exception_and_bindings"] = all(
            key in failure
            for key in ("stage", "progress", "access", "exception", "bindings")
        )
        durable_checks["failure_is_non_authorizing"] = bool(
            not failure["passed"]
            and failure["decision"] == "do_not_authorize_reward_planning"
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

    source = (PROJECT_ROOT / "python/evaluate_v65r3_eig.py").read_text()
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
    durable_checks["registered_zero_branch_count_is_nine"] = (
        '"expected_exact_zero_identity_branch_count": 9' in source
        and 'access["exact_zero_identity_branch_count"] != 9' in source
    )
    durable_checks["V65r3_inference_imported"] = (
        "from v65r3_smc2_eig import smc2_inference" in source
    )
    durable_ok = all(durable_checks.values())
    if not durable_ok:
        errors.append("V65r3 durable attempt/result/failure protocol audit failed")

    original_evaluator_lock = json.loads(
        (PROJECT_ROOT / "configs/v65r1-evaluation-implementation-lock.json").read_text()
    )
    original_evaluator_audit = json.loads(
        (PROJECT_ROOT / original_evaluator_lock["evaluation_implementation_audit"]).read_text()
    )
    inherited_audits_ok = bool(
        original_evaluator_audit["mutation_audit"]["kill_rate"] == 1.0
        and original_evaluator_audit["mutation_audit"]["registered"] == 32
        and original_evaluator_audit["analytic_fixtures"]["pass_rate"] == 1.0
        and implementation_audit_raw["mutation_audit"]["kill_rate"] == 1.0
        and implementation_audit_raw["mutation_audit"]["registered"] == 16
        and implementation_audit_raw["analytic_fixtures"]["pass_rate"] == 1.0
    )
    if not inherited_audits_ok:
        errors.append("inherited evaluator or V65r3 implementation audit is incomplete")

    mutation_checks = {
        **durable_checks,
        "inherited_valid_aggregate_passes": aggregate_ok,
        "inherited_evaluator_32_mutants_killed": original_evaluator_audit[
            "mutation_audit"
        ]["killed"]
        == 32,
        "V65r3_implementation_16_mutants_killed": implementation_audit_raw[
            "mutation_audit"
        ]["killed"]
        == 16,
        "evaluation_directory_absent": not evaluation_dir.exists(),
    }
    mutation_checks = {key: bool(value) for key, value in mutation_checks.items()}
    mutation_ok = all(mutation_checks.values())

    checks = {
        "implementation_binding_and_evaluator_only_authorization": implementation_ok,
        "immutable_subset_binding": subset_binding_ok,
        "valid_inherited_aggregate_fixture": aggregate_ok,
        "durable_one_shot_protocol": durable_ok,
        "inherited_evaluator_and_implementation_audits": inherited_audits_ok,
        "all_registered_durable_and_inherited_checks_pass": mutation_ok,
        "evaluation_directory_absent": not evaluation_dir.exists(),
    }
    audit = {
        "schema_version": "65r3",
        "experiment": "v65r3_evaluator_audit",
        "passed": not errors and all(checks.values()),
        "decision": (
            "freeze_v65r3_evaluator_and_authorize_one_immutable_evaluation"
            if not errors and all(checks.values())
            else "reject_v65r3_evaluator"
        ),
        "errors": errors,
        "checks": checks,
        "mutation_audit": {
            "registered_durable_and_binding_checks": len(mutation_checks),
            "passed_durable_and_binding_checks": sum(mutation_checks.values()),
            "pass_rate": float(np.mean(list(mutation_checks.values()))),
            "checks": mutation_checks,
            "inherited_evaluator_mutants_killed": 32,
            "inherited_implementation_mutants_killed": 16,
            "combined_registered_checks_and_mutants": len(mutation_checks) + 48,
            "combined_passed_or_killed": sum(mutation_checks.values()) + 48,
        },
        "access": {
            "sealed_public_records_loaded": 0,
            "sealed_candidate_EIG_scores": 0,
            "synthetic_record_budget_rows": len(rows),
            "V65r1_evaluation_reruns": 0,
            "V65r2_evaluation_attempts": 0,
            "V65r3_evaluation_attempts": 0,
            "truth_fields_accessed": 0,
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
        "python/evaluate_v65r3_eig.py",
        "python/test_v65r3_evaluator.py",
        "python/audit_and_freeze_v65r3_evaluator.py",
        "python/evaluate_v65r1_eig.py",
        "python/v65r3_smc2_eig.py",
        "python/v65r2_smc2_eig.py",
        "python/v65_smc2_eig.py",
        "python/v65_scalar_reference.py",
        "python/v64_external_eig.py",
        "python/v62_external_pomdp.py",
    )
    lock = {
        "schema_version": "65r3",
        "experiment": "v65r3_evaluation_implementation_lock",
        "implementation_lock": str(implementation_path.relative_to(PROJECT_ROOT)),
        "implementation_lock_sha256": file_sha256(implementation_path),
        "subset_seal": str(subset_seal_path.relative_to(PROJECT_ROOT)),
        "subset_seal_sha256": file_sha256(subset_seal_path),
        "inherited_v65r1_evaluator_lock": "configs/v65r1-evaluation-implementation-lock.json",
        "inherited_v65r1_evaluator_lock_sha256": file_sha256(
            PROJECT_ROOT / "configs/v65r1-evaluation-implementation-lock.json"
        ),
        "evaluator_audit": str(audit_path.relative_to(PROJECT_ROOT)),
        "evaluator_audit_sha256": file_sha256(audit_path),
        "source_sha256": {
            relative: file_sha256(PROJECT_ROOT / relative) for relative in sources
        },
        "authorization": {
            "modify_or_rerun_v65r1": False,
            "modify_or_continue_v65r2": False,
            "modify_v65r3_design_implementation_or_evaluator": False,
            "run_one_immutable_evaluation": True,
            "run_additional_evaluation": False,
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
                "durable_checks": len(durable_checks),
                "combined_checks_and_mutants": audit["mutation_audit"][
                    "combined_registered_checks_and_mutants"
                ],
                "evaluation_directory_absent": not evaluation_dir.exists(),
                "lock": str(output_path.relative_to(PROJECT_ROOT)),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
