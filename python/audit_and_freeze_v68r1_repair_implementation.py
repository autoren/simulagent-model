#!/usr/bin/env python3
"""Synthetic audit and lock for the V68r1 off-support repair."""
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
    design_path = PROJECT_ROOT / "configs/v68r1-development-repair-design-lock.json"
    implementation_path = PROJECT_ROOT / "python/v68r1_posterior_sampling.py"
    tests_path = PROJECT_ROOT / "python/test_v68r1_posterior_sampling.py"
    auditor_path = PROJECT_ROOT / "python/audit_and_freeze_v68r1_repair_implementation.py"
    audit_path = PROJECT_ROOT / "outputs/v68r1-development-screening/implementation-audit.json"
    lock_path = PROJECT_ROOT / "configs/v68r1-development-implementation-lock.json"
    if lock_path.exists():
        raise RuntimeError("V68r1 repair implementation already frozen")
    design = json.loads(design_path.read_text())
    payload = {key: value for key, value in design.items() if key != "lock_payload_sha256"}
    errors: list[str] = []
    design_ok = bool(
        payload_hash(payload) == design["lock_payload_sha256"]
        and design["authorization"]["write_and_audit_off_support_repair"]
        and not design["authorization"]["write_and_audit_repaired_evaluator"]
        and not design["authorization"]["run_repaired_development_screen"]
        and not design["authorization"]["score_confirmatory_models"]
    )
    if not design_ok:
        errors.append("V68r1 repair design lock or implementation-only authorization failed")

    completed = subprocess.run(
        [
            sys.executable, "-m", "unittest", "discover", "-s", "python",
            "-p", "test_v68r1_posterior_sampling.py",
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    combined = completed.stdout + completed.stderr
    tests_ok = completed.returncode == 0 and "Ran 5 tests" in combined
    if not tests_ok:
        errors.append("V68r1 posterior-sampling repair tests failed")

    source = implementation_path.read_text()
    source_checks = {
        "strict_exact_zero_trigger": "point_probability > 0.0" in source,
        "fixed_first_canonical_fallback": (
            "fallback_action = int(kernel.canonical_actions[0])" in source
            and "fallback_action != exact_kernel.canonical_actions[0]" in source
        ),
        "fallback_repeats_for_all_remaining_actions": (
            "(fallback_action,) * (horizon - 1)" in source
        ),
        "sampled_point_model_persists_on_supported_histories": (
            "point_step[\"posteriors\"][observation]" in source
        ),
        "no_model_resampling": '"off_support_model_resampling": False' in source,
        "no_epsilon_smoothing": '"epsilon_smoothing": False' in source,
        "entry_probability_diagnostic": "expected_off_support_entry_probability" in source,
        "branch_count_diagnostic": "off_support_branch_count" in source,
        "unchanged_exact_infrastructure_not_edited": (
            design["unchanged_artifacts"]["implementation"]["sha256"]
            == file_sha256(
                PROJECT_ROOT / design["unchanged_artifacts"]["implementation"]["path"]
            )
        ),
    }
    source_ok = all(source_checks.values())
    if not source_ok:
        errors.append("V68r1 implementation differs from the frozen totalization semantics")

    repaired_evaluator_absent = not any(
        (PROJECT_ROOT / path).exists()
        for path in (
            "python/evaluate_v68r1_development_screen.py",
            "configs/v68r1-development-evaluator-lock.json",
            "outputs/v68r1-development-screening/evaluation",
        )
    )
    if not repaired_evaluator_absent:
        errors.append("V68r1 repaired evaluator exists before repair implementation lock")

    checks = {
        "repair_design_binding_and_implementation_only_authorization": design_ok,
        "five_synthetic_union_support_tests": tests_ok,
        "frozen_totalization_source_semantics": source_ok,
        "repaired_evaluator_and_attempt_absent": repaired_evaluator_absent,
    }
    audit = {
        "schema_version": "68r1-development-screening",
        "experiment": "v68r1_repair_implementation_audit",
        "passed": not errors and all(checks.values()),
        "decision": (
            "freeze_off_support_implementation_and_authorize_repaired_evaluator_only"
            if not errors
            else "reject_v68r1_repair_implementation"
        ),
        "errors": errors,
        "checks": checks,
        "source_checks": source_checks,
        "access": {
            "synthetic_support_mismatch_fixtures": 5,
            "additional_development_records_evaluated": 0,
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
        "schema_version": "68r1-development-screening",
        "experiment": "v68r1_development_implementation_lock",
        "repair_design_lock": str(design_path.relative_to(PROJECT_ROOT)),
        "repair_design_lock_sha256": file_sha256(design_path),
        "implementation": str(implementation_path.relative_to(PROJECT_ROOT)),
        "implementation_sha256": file_sha256(implementation_path),
        "tests": str(tests_path.relative_to(PROJECT_ROOT)),
        "tests_sha256": file_sha256(tests_path),
        "implementation_audit": str(audit_path.relative_to(PROJECT_ROOT)),
        "implementation_audit_sha256": file_sha256(audit_path),
        "implementation_auditor": str(auditor_path.relative_to(PROJECT_ROOT)),
        "implementation_auditor_sha256": file_sha256(auditor_path),
        "authorization": {
            "modify_repair_design_or_implementation": False,
            "write_and_audit_repaired_durable_evaluator": True,
            "run_repaired_development_screen": False,
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
