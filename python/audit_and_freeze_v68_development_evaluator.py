#!/usr/bin/env python3
"""Audit and lock the one-shot V68 development-screen evaluator."""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from typing import Any

from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT


def payload_hash(value: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def main() -> None:
    seal_path = PROJECT_ROOT / "configs/v68-development-census-seal.json"
    evaluator_path = PROJECT_ROOT / "python/evaluate_v68_development_screen.py"
    tests_path = PROJECT_ROOT / "python/test_v68_development_evaluator.py"
    auditor_path = PROJECT_ROOT / "python/audit_and_freeze_v68_development_evaluator.py"
    audit_path = PROJECT_ROOT / "outputs/v68-development-screening/evaluator-audit.json"
    lock_path = PROJECT_ROOT / "configs/v68-development-evaluator-lock.json"
    if lock_path.exists():
        raise RuntimeError("V68 development evaluator already frozen")
    seal = json.loads(seal_path.read_text())
    errors: list[str] = []
    seal_payload = {key: value for key, value in seal.items() if key != "lock_payload_sha256"}
    seal_ok = bool(
        payload_hash(seal_payload) == seal["lock_payload_sha256"]
        and seal["authorization"]["write_and_audit_durable_development_evaluator"]
        and not seal["authorization"]["run_development_screen"]
        and not seal["authorization"]["score_confirmatory_models"]
        and seal["record_count"] == 59
        and seal["selection_rejection_or_replacement_count"] == 0
        and file_sha256(PROJECT_ROOT / seal["census"]) == seal["census_sha256"]
    )
    if not seal_ok:
        errors.append("V68 census seal or evaluator-only authorization failed")

    completed = subprocess.run(
        [
            sys.executable, "-m", "unittest", "discover", "-s", "python",
            "-p", "test_v68_development_evaluator.py",
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    combined = completed.stdout + completed.stderr
    tests_ok = completed.returncode == 0 and "Ran 4 tests" in combined
    if not tests_ok:
        errors.append("V68 development evaluator tests failed")

    source = evaluator_path.read_text()
    source_checks = {
        "atomic_attempt_before_record_evaluation": (
            source.index("attempt_path.write_text") < source.index("rows = []")
        ),
        "one_shot_existing_output_rejection": "has already been attempted" in source,
        "sealed_census_hash_check": "sealed census hash mismatch" in source,
        "evaluator_self_hash_check": "evaluator source hash mismatch" in source,
        "confirmatory_firewall": "score_confirmatory_models" in source,
        "complete_five_control_set": all(
            token in source
            for token in (
                '"map"', '"posterior_sampling"', '"open_loop"',
                '"myopic_reward"', '"information_only"',
            )
        ),
        "primary_and_convergence_quadrature": all(
            token in source for token in ("primaryQuadratureNodes", "convergenceQuadratureNodes")
        ),
        "no_hardcoded_confirmatory_filename": not any(
            name in source
            for name in (
                "cheese.95.POMDP", "fully_observable_tmaze2.POMDP", "hallway.POMDP",
                "heavenhell.POMDP", "network.POMDP", "shuttle.POMDP", "paint.POMDP",
            )
        ),
    }
    source_ok = all(source_checks.values())
    if not source_ok:
        errors.append("V68 evaluator durability, controls, convergence, or firewall source checks failed")

    design = json.loads(PROJECT_ROOT.joinpath(seal["development_design_lock"]).read_text())
    gates = design["config_payload"]["gates"]
    gates_ok = bool(
        len(gates) == 19
        and gates["minimumCompletedRecordFraction"] == 1.0
        and gates["minimumExactBAMinusMAPRootActionDisagreementRecords"] == 3
        and gates["minimumExactBAMinusMAPMaterialRegretRecords"] == 2
        and gates["minimumMaximumNormalizedMAPRegret"] == 0.01
        and gates["minimumExactBAMinusOpenLoopMaterialRegretRecords"] == 2
        and gates["minimumExactBAMinusPosteriorSamplingMaterialRegretRecords"] == 1
        and gates["maximumConfirmatoryModelsScored"] == 0
        and gates["maximumRecordSelectionOrRejectionCount"] == 0
    )
    if not gates_ok:
        errors.append("V68 frozen gate count or material separation gates differ")

    evaluation_dir = PROJECT_ROOT / "outputs/v68-development-screening/evaluation"
    evaluation_absent = not evaluation_dir.exists()
    if not evaluation_absent:
        errors.append("V68 development evaluation exists before evaluator lock")

    checks = {
        "sealed_complete_census_and_evaluator_only_authorization": seal_ok,
        "four_synthetic_evaluator_and_gate_tests": tests_ok,
        "durable_one_shot_source_controls_and_firewall": source_ok,
        "nineteen_frozen_noncompensatory_gates": gates_ok,
        "evaluation_absent_before_evaluator_lock": evaluation_absent,
    }
    audit = {
        "schema_version": "68-development-screening",
        "experiment": "v68_development_evaluator_audit",
        "passed": not errors and all(checks.values()),
        "decision": (
            "freeze_durable_evaluator_and_authorize_one_development_screen"
            if not errors
            else "reject_v68_development_evaluator"
        ),
        "errors": errors,
        "checks": checks,
        "source_checks": source_checks,
        "access": {
            "synthetic_evaluator_records": 4,
            "sealed_development_records_evaluated": 0,
            "confirmatory_models_scored": 0,
            "SMC2_runs": 0,
            "human_records": 0,
            "model_forward_passes": 0,
            "adapter_training_runs": 0,
        },
    }
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    if not audit["passed"]:
        print(json.dumps(audit, indent=2, sort_keys=True))
        raise SystemExit(1)
    lock = {
        "schema_version": "68-development-screening",
        "experiment": "v68_development_evaluator_lock",
        "development_census_seal": str(seal_path.relative_to(PROJECT_ROOT)),
        "development_census_seal_sha256": file_sha256(seal_path),
        "evaluator": str(evaluator_path.relative_to(PROJECT_ROOT)),
        "evaluator_sha256": file_sha256(evaluator_path),
        "evaluator_tests": str(tests_path.relative_to(PROJECT_ROOT)),
        "evaluator_tests_sha256": file_sha256(tests_path),
        "evaluator_audit": str(audit_path.relative_to(PROJECT_ROOT)),
        "evaluator_audit_sha256": file_sha256(audit_path),
        "evaluator_auditor": str(auditor_path.relative_to(PROJECT_ROOT)),
        "evaluator_auditor_sha256": file_sha256(auditor_path),
        "attempt_path": "outputs/v68-development-screening/evaluation/attempt.json",
        "expected_attempt_number": 1,
        "expected_records": 59,
        "expected_confirmatory_models_scored": 0,
        "authorization": {
            "modify_design_implementation_census_or_evaluator": False,
            "run_development_screen_once": True,
            "score_confirmatory_models": False,
            "run_SMC2": False,
            "human_data": False,
            "model_access": False,
            "adapter_training": False,
        },
    }
    lock["lock_payload_sha256"] = payload_hash(lock)
    lock_path.write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n")
    print(json.dumps(audit, indent=2, sort_keys=True))
    print(json.dumps({"lock": str(lock_path), "sha256": file_sha256(lock_path)}, indent=2))


if __name__ == "__main__":
    main()
