#!/usr/bin/env python3
"""Audit and lock the V69 exact dominant-remapping infrastructure."""
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
    design_path = PROJECT_ROOT / "configs/v69-development-design-lock.json"
    implementation_path = PROJECT_ROOT / "python/v69_dominant_remapping.py"
    tests_path = PROJECT_ROOT / "python/test_v69_dominant_remapping.py"
    auditor_path = PROJECT_ROOT / "python/audit_and_freeze_v69_implementation.py"
    audit_path = PROJECT_ROOT / "outputs/v69-development-screening/implementation-audit.json"
    lock_path = PROJECT_ROOT / "configs/v69-development-implementation-lock.json"
    if lock_path.exists():
        raise RuntimeError("V69 implementation already frozen")
    design = json.loads(design_path.read_text())
    payload = {key: value for key, value in design.items() if key != "lock_payload_sha256"}
    errors: list[str] = []
    design_ok = bool(
        payload_hash(payload) == design["lock_payload_sha256"]
        and design["authorization"]["write_and_audit_exact_infrastructure"]
        and not design["authorization"]["construct_development_census"]
        and not design["authorization"]["run_development_screen"]
        and not design["authorization"]["score_confirmatory_models"]
    )
    if not design_ok:
        errors.append("V69 design lock or implementation-only authorization failed")

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "unittest",
            "discover",
            "-s",
            "python",
            "-p",
            "test_v69_dominant_remapping.py",
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    combined = completed.stdout + completed.stderr
    tests_ok = completed.returncode == 0 and "Ran 5 tests" in combined
    if not tests_ok:
        errors.append(f"V69 infrastructure tests failed: {combined[-1200:]}")

    source = implementation_path.read_text()
    source_checks = {
        "theta_weights_remapped_transition": (
            "value * model.transition[permutation[action]]" in source
        ),
        "one_minus_theta_weights_nominal_transition": (
            "(1.0 - value) * model.transition[action]" in source
        ),
        "two_equal_identity_priors": (
            "np.concatenate([0.5 * theta_weights, 0.5 * theta_weights])" in source
        ),
        "frozen_identity_names": (
            "dominant_forward_cycle_remapping" in source
            and "dominant_backward_cycle_remapping" in source
        ),
        "reuses_exact_static_kernel": "StaticKernel(" in source,
        "reuses_frozen_cycle_constructor": "cycle_permutations" in source,
    }
    source_ok = all(source_checks.values())
    if not source_ok:
        errors.append("V69 implementation does not match dominant-remapping algebra")

    prior_ok = all(
        file_sha256(PROJECT_ROOT / artifact["path"]) == artifact["sha256"]
        for artifact in (
            json.loads(
                (PROJECT_ROOT / "configs/v68r2-development-outcome-lock.json").read_text()
            )["authorization"]
            and [
                {
                    "path": "python/v68_multi_environment_exact.py",
                    "sha256": json.loads(
                        (PROJECT_ROOT / "configs/v68-development-implementation-lock.json").read_text()
                    )["implementation_sha256"],
                },
                {
                    "path": "python/v68r2_point_model_controls.py",
                    "sha256": json.loads(
                        (PROJECT_ROOT / "configs/v68r2-development-implementation-lock.json").read_text()
                    )["implementation_sha256"],
                },
            ]
        )
    )
    if not prior_ok:
        errors.append("locked exact or totalized control infrastructure drifted")

    downstream_absent = not any(
        (PROJECT_ROOT / path).exists()
        for path in (
            "configs/v69-development-census-seal.json",
            "python/evaluate_v69_development_screen.py",
            "configs/v69-development-evaluator-lock.json",
            "outputs/v69-development-screening/evaluation",
        )
    )
    if not downstream_absent:
        errors.append("V69 census or evaluator exists before implementation lock")

    checks = {
        "design_binding_and_implementation_only_authorization": design_ok,
        "five_synthetic_exact_infrastructure_tests": tests_ok,
        "frozen_dominant_remapping_source_semantics": source_ok,
        "locked_prior_exact_and_point_control_infrastructure": prior_ok,
        "census_and_evaluator_absent": downstream_absent,
    }
    audit = {
        "schema_version": "69-development-screening",
        "experiment": "v69_implementation_audit",
        "passed": not errors and all(checks.values()),
        "decision": (
            "freeze_exact_infrastructure_and_authorize_census_only"
            if not errors
            else "reject_v69_implementation"
        ),
        "errors": errors,
        "checks": checks,
        "source_checks": source_checks,
        "access": {
            "synthetic_fixtures": 5,
            "development_records_evaluated": 0,
            "confirmatory_models_scored": 0,
        },
    }
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    if not audit["passed"]:
        print(json.dumps(audit, indent=2, sort_keys=True))
        raise SystemExit(1)
    lock = {
        "schema_version": "69-development-screening",
        "experiment": "v69_development_implementation_lock",
        "development_design_lock": str(design_path.relative_to(PROJECT_ROOT)),
        "development_design_lock_sha256": file_sha256(design_path),
        "implementation": str(implementation_path.relative_to(PROJECT_ROOT)),
        "implementation_sha256": file_sha256(implementation_path),
        "tests": str(tests_path.relative_to(PROJECT_ROOT)),
        "tests_sha256": file_sha256(tests_path),
        "implementation_audit": str(audit_path.relative_to(PROJECT_ROOT)),
        "implementation_audit_sha256": file_sha256(audit_path),
        "implementation_auditor": str(auditor_path.relative_to(PROJECT_ROOT)),
        "implementation_auditor_sha256": file_sha256(auditor_path),
        "authorization": {
            "modify_design_or_implementation": False,
            "construct_and_seal_development_census": True,
            "write_and_audit_development_evaluator": False,
            "run_development_screen": False,
            "score_confirmatory_models": False,
        },
    }
    lock["lock_payload_sha256"] = payload_hash(lock)
    lock_path.write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n")
    print(json.dumps(audit, indent=2, sort_keys=True))
    print(json.dumps({"lock": str(lock_path), "sha256": file_sha256(lock_path)}, indent=2))


if __name__ == "__main__":
    main()
