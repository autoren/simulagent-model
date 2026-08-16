#!/usr/bin/env python3
"""Durable one-shot V65r3 evaluation over the unchanged sealed V65r1 subset."""
from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Sequence

from evaluate_v65r1_eig import (
    aggregate_evaluation,
    compare_record_budget,
    exact_reference,
    read_jsonl,
)
from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT
from v64_external_eig import load_family
from v65r3_smc2_eig import smc2_inference


TERMINAL_NAMES = (
    "attempt.json",
    "result.json",
    "failure.json",
    "record-budget-cells.jsonl",
)


def aggregate_implementation_audit(audit: dict[str, Any]) -> dict[str, Any]:
    """Expose the frozen V65r3 shared-stream check under V65r1's inherited key."""
    result = copy.deepcopy(audit)
    checks = result["mutation_audit"]["checks"]
    if "share_inner_random_streams" not in checks:
        raise RuntimeError("V65r3 implementation audit omits shared-stream detection")
    checks["share_inner_streams_across_outer_particles"] = bool(
        checks["share_inner_random_streams"]
    )
    return result


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    if temporary.exists():
        raise RuntimeError(f"V65r3 stale atomic-write temporary exists: {temporary.name}")
    try:
        with temporary.open("x") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    atomic_write_text(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def atomic_write_jsonl(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    atomic_write_text(
        path,
        "".join(
            json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
            for row in rows
        ),
    )


def reserve_attempt(output_dir: Path, marker: dict[str, Any]) -> Path:
    existing = [name for name in TERMINAL_NAMES if (output_dir / name).exists()]
    if existing:
        raise RuntimeError(
            "V65r3 one-shot evaluation already attempted or materialized: "
            + ",".join(existing)
        )
    attempt_path = output_dir / "attempt.json"
    atomic_write_json(attempt_path, marker)
    return attempt_path


def failure_payload(
    *,
    lock_path: Path,
    attempt_path: Path,
    stage: str,
    progress: dict[str, int],
    access: dict[str, int],
    error: BaseException,
) -> dict[str, Any]:
    return {
        "schema_version": "65r3",
        "experiment": "v65r3_pooled_smc2_eig_portability",
        "passed": False,
        "status": "terminal_exception",
        "decision": "do_not_authorize_reward_planning",
        "one_shot_authorization_consumed": True,
        "stage": stage,
        "progress": progress,
        "access": access,
        "exception": {
            "type": type(error).__name__,
            "message": str(error),
            "traceback": traceback.format_exc(),
        },
        "bindings": {
            "evaluation_implementation_lock": str(lock_path.relative_to(PROJECT_ROOT)),
            "evaluation_implementation_lock_sha256": file_sha256(lock_path),
            "attempt": str(attempt_path.relative_to(PROJECT_ROOT)),
            "attempt_sha256": file_sha256(attempt_path),
        },
        "claim_boundary": {
            "accuracy_gates_evaluable": False,
            "V65r3_rerun_authorized": False,
            "reward_planning_authorized": False,
        },
    }


def run_evaluation(lock_path: Path, output_dir: Path) -> dict[str, Any]:
    lock = json.loads(lock_path.read_text())
    if not lock["authorization"]["run_one_immutable_evaluation"]:
        raise RuntimeError("V65r3 evaluator lock does not authorize evaluation")
    if lock["authorization"]["run_additional_evaluation"]:
        raise RuntimeError("V65r3 evaluator lock improperly authorizes additional evaluation")
    for relative, digest in lock["source_sha256"].items():
        if file_sha256(PROJECT_ROOT / relative) != digest:
            raise RuntimeError(f"frozen V65r3 evaluator or dependency changed: {relative}")
    subset_seal_path = PROJECT_ROOT / lock["subset_seal"]
    if file_sha256(subset_seal_path) != lock["subset_seal_sha256"]:
        raise RuntimeError("V65r3 sealed subset changed before attempt")
    implementation_path = PROJECT_ROOT / lock["implementation_lock"]
    if file_sha256(implementation_path) != lock["implementation_lock_sha256"]:
        raise RuntimeError("V65r3 implementation lock changed before attempt")

    marker = {
        "schema_version": "65r3",
        "experiment": "v65r3_immutable_evaluation_attempt",
        "logical_evaluation_attempt": 1,
        "one_shot_authorization_consumed": True,
        "evaluation_implementation_lock": str(lock_path.relative_to(PROJECT_ROOT)),
        "evaluation_implementation_lock_sha256": file_sha256(lock_path),
        "subset_seal_sha256": lock["subset_seal_sha256"],
        "implementation_lock_sha256": lock["implementation_lock_sha256"],
    }
    attempt_path = reserve_attempt(output_dir, marker)
    stage = "attempt_reserved_before_subset_load"
    progress = {
        "sealed_records_loaded": 0,
        "exact_references_completed": 0,
        "SMC2_repeat_cells_completed": 0,
        "record_budget_rows_completed": 0,
    }
    access = {
        "logical_evaluation_attempts": 1,
        "subset_public_records_loaded": 0,
        "v64_source_public_records_loaded_during_evaluation": 0,
        "v64_selection_audit_records_loaded": 0,
        "v64_evaluation_records_loaded": 0,
        "truth_field_access_count": 0,
        "realized_outcome_access_before_selection_count": 0,
        "candidate_omission_count": 0,
        "tie_break_violation_count": 0,
        "random_stream_collision_count": 0,
        "human_record_access_count": 0,
        "model_forward_pass_count": 0,
        "adapter_training_run_count": 0,
        "V65r1_evaluation_reruns": 0,
        "V65r2_evaluation_attempts": 0,
        "exact_zero_identity_branch_count": 0,
    }
    started = time.perf_counter()
    try:
        stage = "load_frozen_configuration_and_subset"
        implementation = json.loads(implementation_path.read_text())
        design_path = PROJECT_ROOT / implementation["design_lock"]
        design = json.loads(design_path.read_text())
        config = design["config_payload"]
        implementation_audit = aggregate_implementation_audit(json.loads(
            (PROJECT_ROOT / implementation["implementation_audit"]).read_text()
        ))
        subset_seal = json.loads(subset_seal_path.read_text())
        subset_path = PROJECT_ROOT / subset_seal["files"]["subset_public"]["path"]
        if file_sha256(subset_path) != subset_seal["files"]["subset_public"]["sha256"]:
            raise RuntimeError("V65r3 public subset changed after seal")
        records = read_jsonl(subset_path)
        progress["sealed_records_loaded"] = len(records)
        access["subset_public_records_loaded"] = len(records)
        family = load_family()
        rows = []
        stage = "record_budget_repeat_inference_and_scoring"
        for record in records:
            exact = exact_reference(family, record)
            progress["exact_references_completed"] += 1
            for budget in config["smcSquared"]["outerThetaParticleBudgets"]:
                repeats = []
                for repeat in range(
                    config["smcSquared"]["independentRepeatsPerBudget"]
                ):
                    inference = smc2_inference(
                        family, record, config, int(budget), repeat
                    )
                    repeats.append(inference)
                    progress["SMC2_repeat_cells_completed"] += 1
                    access["random_stream_collision_count"] += int(
                        inference["diagnostics"]["random_stream_collision_count"]
                    )
                    access["exact_zero_identity_branch_count"] += int(
                        inference["diagnostics"]["exact_zero_identity_count"]
                    )
                row = compare_record_budget(
                    family, record, exact, repeats, int(budget)
                )
                row["support_diagnostics"] = [
                    {
                        "repeat": int(inference["repeat"]),
                        "exact_support_by_identity": inference["diagnostics"][
                            "exact_support_by_identity"
                        ],
                        "support_extinction_tick_by_identity": inference["diagnostics"][
                            "support_extinction_tick_by_identity"
                        ],
                        "exact_zero_identity_count": inference["diagnostics"][
                            "exact_zero_identity_count"
                        ],
                    }
                    for inference in repeats
                ]
                rows.append(row)
                progress["record_budget_rows_completed"] += 1

        access["candidate_omission_count"] = int(
            sum(
                row["candidate_count"]
                != len(config["approximateAcquisition"]["candidateOrder"])
                or row["candidate_order"]
                != config["approximateAcquisition"]["candidateOrder"]
                for row in rows
            )
        )
        access["tie_break_violation_count"] = int(
            sum(not row["tie_break_valid"] for row in rows)
        )
        stage = "aggregate_original_noncompensatory_gates"
        result = aggregate_evaluation(rows, config, implementation_audit, access)
        result["schema_version"] = "65r3"
        result["experiment"] = "v65r3_pooled_smc2_eig_portability"
        result["runtime_seconds"] = time.perf_counter() - started
        result["bindings"] = {
            "evaluation_implementation_lock": str(lock_path.relative_to(PROJECT_ROOT)),
            "evaluation_implementation_lock_sha256": file_sha256(lock_path),
            "implementation_lock": str(implementation_path.relative_to(PROJECT_ROOT)),
            "implementation_lock_sha256": file_sha256(implementation_path),
            "subset_seal": str(subset_seal_path.relative_to(PROJECT_ROOT)),
            "subset_seal_sha256": file_sha256(subset_seal_path),
            "attempt": str(attempt_path.relative_to(PROJECT_ROOT)),
            "attempt_sha256": file_sha256(attempt_path),
        }
        result["repair_diagnostics"] = {
            "exact_zero_identity_branch_count": access[
                "exact_zero_identity_branch_count"
            ],
            "expected_exact_zero_identity_branch_count": 9,
            "positive_support_particle_extinction_count": 0,
        }
        if access["exact_zero_identity_branch_count"] != 9:
            result["gate_checks"]["registered_exact_zero_identity_branch_count"] = False
            result["failed_gates"].append(
                "registered_exact_zero_identity_branch_count"
            )
            result["passed"] = False
            result["decision"] = "do_not_authorize_reward_planning"
        else:
            result["gate_checks"]["registered_exact_zero_identity_branch_count"] = True
        stage = "atomically_write_raw_cells_and_result"
        raw_path = output_dir / "record-budget-cells.jsonl"
        atomic_write_jsonl(raw_path, rows)
        result["record_budget_cells"] = str(raw_path.relative_to(PROJECT_ROOT))
        result["record_budget_cells_sha256"] = file_sha256(raw_path)
        atomic_write_json(output_dir / "result.json", result)
        return result
    except BaseException as error:
        if isinstance(error, (KeyboardInterrupt, SystemExit)):
            raise
        failure = failure_payload(
            lock_path=lock_path,
            attempt_path=attempt_path,
            stage=stage,
            progress=progress,
            access=access,
            error=error,
        )
        atomic_write_json(output_dir / "failure.json", failure)
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--lock", default="configs/v65r3-evaluation-implementation-lock.json"
    )
    parser.add_argument(
        "--output-dir", default="outputs/v65r3-synthetic-only-implementation/evaluation"
    )
    args = parser.parse_args()
    lock_path = (PROJECT_ROOT / args.lock).resolve()
    output_dir = (PROJECT_ROOT / args.output_dir).resolve()
    try:
        result = run_evaluation(lock_path, output_dir)
    except Exception as error:
        print(
            json.dumps(
                {
                    "passed": False,
                    "decision": "do_not_authorize_reward_planning",
                    "exception": {"type": type(error).__name__, "message": str(error)},
                    "failure": str((output_dir / "failure.json").relative_to(PROJECT_ROOT)),
                },
                indent=2,
                sort_keys=True,
            )
        )
        raise SystemExit(1) from error
    print(
        json.dumps(
            {
                "passed": result["passed"],
                "decision": result["decision"],
                "failed_gates": result["failed_gates"],
                "by_budget": result["by_budget"],
                "controls": result["controls"],
                "repair_diagnostics": result["repair_diagnostics"],
                "runtime_seconds": result["runtime_seconds"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
