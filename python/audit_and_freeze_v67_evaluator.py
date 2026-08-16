#!/usr/bin/env python3
"""Audit and freeze the durable V67 one-shot evaluator."""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from typing import Any

from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT
from v67_verification import canonical_json, storm_version


def payload_hash(value: dict[str, Any]) -> str:
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


def main() -> None:
    seal_path = PROJECT_ROOT / "configs/v67-verification-bundle-seal.json"
    evaluator_path = PROJECT_ROOT / "python/evaluate_v67_verification.py"
    tests_path = PROJECT_ROOT / "python/test_v67_evaluator.py"
    auditor_path = PROJECT_ROOT / "python/audit_and_freeze_v67_evaluator.py"
    audit_path = PROJECT_ROOT / "outputs/v67-independent-bounded-policy-verification/evaluator-audit.json"
    lock_path = PROJECT_ROOT / "configs/v67-evaluation-implementation-lock.json"
    if lock_path.exists():
        raise RuntimeError("V67 evaluator already frozen")
    seal = json.loads(seal_path.read_text())
    seal_payload = {key: value for key, value in seal.items() if key != "lock_payload_sha256"}
    errors: list[str] = []
    seal_ok = bool(
        payload_hash(seal_payload) == seal["lock_payload_sha256"]
        and seal["authorization"]["write_and_audit_durable_evaluator"]
        and not seal["authorization"]["run_verification"]
        and seal["policy_count"] == 96
        and file_sha256(PROJECT_ROOT / seal["bundle_manifest"])
        == seal["bundle_manifest_sha256"]
        and file_sha256(PROJECT_ROOT / seal["bundle_audit"])
        == seal["bundle_audit_sha256"]
    )
    if not seal_ok:
        errors.append("V67 bundle seal or evaluator-only authorization failed")

    source = evaluator_path.read_text()
    attempt_order_ok = bool(
        source.index("reserve_attempt(attempt_path, attempt)")
        < source.index("manifest = json.loads(manifest_path.read_text())")
        < source.index("rows.append(verify_policy_directory(")
    )
    protections_ok = bool(
        "os.O_EXCL" in source
        and "atomic_json(failure_path" in source
        and "validate_manifest_row_files" in source
        and "storm_version() != \"1.13.0\"" in source
        and "source_result_mutation_count" in source
        and "unexpected_attempt_count=0" in source
        and attempt_order_ok
    )
    if not protections_ok:
        errors.append("V67 durable attempt, seal, tool, or terminal-failure protections are incomplete")

    completed = subprocess.run(
        [sys.executable, str(tests_path)], cwd=PROJECT_ROOT,
        env={**os.environ, "PYTHONPATH": str(PROJECT_ROOT / "python")},
        capture_output=True, text=True,
    )
    tests_run = 0
    for line in (completed.stdout + completed.stderr).splitlines():
        if line.startswith("Ran "):
            tests_run = int(line.split()[1])
    tests_ok = completed.returncode == 0 and tests_run == 8
    if not tests_ok:
        errors.append("V67 evaluator synthetic tests failed")

    verification_root = (
        PROJECT_ROOT / "outputs/v67-independent-bounded-policy-verification/verification"
    )
    verification_absent = not verification_root.exists()
    if not verification_absent:
        errors.append("V67 verification artifacts exist before evaluator lock")
    tool_ok = storm_version() == "1.13.0"
    if not tool_ok:
        errors.append("V67 Storm version mismatch")
    audit = {
        "schema_version": "67",
        "experiment": "v67_evaluation_implementation_audit",
        "passed": not errors,
        "decision": "freeze_evaluator_and_authorize_exactly_one_verification" if not errors else "reject_evaluator",
        "errors": errors,
        "checks": {
            "sealed_bundle_and_evaluator_only_authorization": seal_ok,
            "attempt_reserved_before_policy_bundle_access": attempt_order_ok,
            "durable_failure_hash_tool_and_attempt_protections": protections_ok,
            "eight_synthetic_evaluator_tests": tests_ok,
            "Storm_version": storm_version(),
            "verification_artifacts_absent": verification_absent,
        },
        "unit_tests": {
            "tests_run": tests_run, "returncode": completed.returncode,
            "stdout": completed.stdout[-4000:], "stderr": completed.stderr[-4000:],
        },
        "access": {
            "sealed_bundle_manifests_loaded": 1,
            "source_V66_policy_models_run_through_Storm": 0,
            "truth_fields": 0,
            "V66_evaluation_reruns": 0,
            "V67_verification_attempts": 0,
            "human_records": 0,
            "model_forward_passes": 0,
            "adapter_training_runs": 0,
        },
    }
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    if errors:
        print(json.dumps(audit, indent=2, sort_keys=True))
        raise SystemExit(1)
    lock = {
        "schema_version": "67",
        "experiment": "v67_evaluation_implementation_lock",
        "bundle_seal": str(seal_path.relative_to(PROJECT_ROOT)),
        "bundle_seal_sha256": file_sha256(seal_path),
        "bundle_manifest": seal["bundle_manifest"],
        "bundle_manifest_sha256": seal["bundle_manifest_sha256"],
        "evaluator": str(evaluator_path.relative_to(PROJECT_ROOT)),
        "evaluator_sha256": file_sha256(evaluator_path),
        "tests": str(tests_path.relative_to(PROJECT_ROOT)),
        "tests_sha256": file_sha256(tests_path),
        "evaluator_auditor": str(auditor_path.relative_to(PROJECT_ROOT)),
        "evaluator_auditor_sha256": file_sha256(auditor_path),
        "evaluator_audit": str(audit_path.relative_to(PROJECT_ROOT)),
        "evaluator_audit_sha256": file_sha256(audit_path),
        "tool_versions": {"Storm": storm_version(), "Python": sys.version.split()[0]},
        "authorization": {
            "modify_or_rerun_v66": False,
            "modify_v67_design_implementation_bundle_or_evaluator": False,
            "run_exactly_one_verification": True,
            "run_additional_verification": False,
            "truth_field_access": False,
            "human_data": False,
            "model_access": False,
            "adapter_training": False,
        },
    }
    lock["lock_payload_sha256"] = payload_hash(lock)
    lock_path.write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n")
    print(json.dumps(audit, indent=2, sort_keys=True))
    print(json.dumps({"evaluation_lock": str(lock_path), "sha256": file_sha256(lock_path)}, indent=2))


if __name__ == "__main__":
    main()
