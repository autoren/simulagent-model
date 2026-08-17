#!/usr/bin/env python3
"""Audit and lock the one-shot V70 confirmatory evaluator."""
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
    reporting_path = PROJECT_ROOT / "configs/v70-confirmatory-reporting-lock.json"
    evaluator_path = PROJECT_ROOT / "python/evaluate_v70_confirmatory.py"
    aggregation_path = PROJECT_ROOT / "python/v70_confirmatory_aggregation.py"
    tests = [
        PROJECT_ROOT / "python/test_v70_confirmatory_aggregation.py",
        PROJECT_ROOT / "python/test_v70_confirmatory_evaluator.py",
    ]
    auditor_path = PROJECT_ROOT / "python/audit_and_freeze_v70_confirmatory_evaluator.py"
    audit_path = PROJECT_ROOT / "outputs/v70-confirmatory/evaluator-audit.json"
    lock_path = PROJECT_ROOT / "configs/v70-confirmatory-evaluator-lock.json"
    if lock_path.exists():
        raise RuntimeError("V70 evaluator already frozen")
    reporting = json.loads(reporting_path.read_text())
    reporting_payload = {
        key: value for key, value in reporting.items() if key != "lock_payload_sha256"
    }
    errors: list[str] = []
    reporting_ok = bool(
        payload_hash(reporting_payload) == reporting["lock_payload_sha256"]
        and reporting["authorization"]["write_and_audit_confirmatory_evaluator"]
        and not reporting["authorization"]["run_confirmatory_outcome"]
        and not reporting["authorization"]["rescore_development_models"]
        and not reporting["authorization"]["drop_or_replace_models"]
    )
    if not reporting_ok:
        errors.append("V70 reporting lock or evaluator-only authorization failed")

    census = json.loads((PROJECT_ROOT / reporting["census_seal"]).read_text())
    design = json.loads(
        (PROJECT_ROOT / census["confirmatory_design_lock"]).read_text()
    )
    family_lock = json.loads(
        (PROJECT_ROOT / design["config_payload"]["familyLock"]).read_text()
    )
    family_path = PROJECT_ROOT / family_lock["implementation"]
    point_lock = json.loads(
        (PROJECT_ROOT / "configs/v68r2-development-implementation-lock.json").read_text()
    )
    point_path = PROJECT_ROOT / point_lock["implementation"]
    record_evaluator_lock = json.loads(
        (PROJECT_ROOT / "configs/v68r2-development-evaluator-lock.json").read_text()
    )
    record_evaluator_path = PROJECT_ROOT / record_evaluator_lock["evaluator"]
    dependencies_ok = bool(
        file_sha256(PROJECT_ROOT / reporting["census_seal"])
        == reporting["census_seal_sha256"]
        and file_sha256(family_path) == family_lock["implementation_sha256"]
        and file_sha256(point_path) == point_lock["implementation_sha256"]
        and file_sha256(record_evaluator_path)
        == record_evaluator_lock["evaluator_sha256"]
    )
    if not dependencies_ok:
        errors.append("V70 census, family, point controls, or record evaluator drifted")

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "unittest",
            "discover",
            "-s",
            "python",
            "-p",
            "test_v70_*.py",
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    combined = completed.stdout + completed.stderr
    tests_ok = completed.returncode == 0 and "Ran 7 tests" in combined
    if not tests_ok:
        errors.append(f"V70 evaluator and aggregation tests failed: {combined[-1600:]}")

    source = evaluator_path.read_text()
    aggregation_source = aggregation_path.read_text()
    source_checks = {
        "uses_locked_totalized_record_evaluator": (
            "from evaluate_v68r2_development_screen import evaluate_record" in source
        ),
        "uses_dominant_remapping_family": "build_dominant_remapping_family" in source,
        "uses_model_level_aggregation": "aggregate_confirmatory_rows" in source,
        "attempt_precedes_record_evaluation": (
            source.index("attempt_path.write_text") < source.index("rows: list")
        ),
        "rows_persist_only_after_complete_loop": (
            source.index("rows_path.write_text") > source.index("V70_model_complete")
        ),
        "separate_confirmatory_output": "outputs/v70-confirmatory/evaluation" in source,
        "development_rescore_firewall": "development_models_rescored" in source,
        "paired_same_record_qualification": (
            'row["exact_ba_map_root_action_disagreement"]' in aggregation_source
            and 'row["normalized_regrets"]["map"] >= threshold' in aggregation_source
        ),
        "fallback_diagnostics_non_gating": (
            "fallback_diagnostics" in aggregation_source
            and "gate_results" in aggregation_source
        ),
        "Tier_B_pair_separate": "Tier_B_cheese_pair" in aggregation_source,
        "no_hardcoded_development_filename": not any(
            name in source
            for name in (
                "4x3_nonterminating.POMDP",
                "tiger-alt-start.POMDP",
                "tmaze2.POMDP",
                "tmaze5.POMDP",
            )
        ),
    }
    source_ok = all(source_checks.values())
    if not source_ok:
        errors.append("V70 evaluator or aggregation expands beyond frozen reporting")
    evaluation_absent = not (
        PROJECT_ROOT / "outputs/v70-confirmatory/evaluation"
    ).exists()
    if not evaluation_absent:
        errors.append("V70 outcome exists before evaluator lock")

    checks = {
        "reporting_binding_and_evaluator_only_authorization": reporting_ok,
        "locked_census_family_point_controls_and_record_evaluator": dependencies_ok,
        "seven_synthetic_evaluator_and_gate_tests": tests_ok,
        "durable_complete_model_level_evaluator_source": source_ok,
        "confirmatory_outcome_absent_before_lock": evaluation_absent,
    }
    audit = {
        "schema_version": "70-confirmatory-multi-environment",
        "experiment": "v70_confirmatory_evaluator_audit",
        "passed": not errors and all(checks.values()),
        "decision": (
            "freeze_evaluator_and_authorize_one_confirmatory_outcome"
            if not errors
            else "reject_v70_confirmatory_evaluator"
        ),
        "errors": errors,
        "checks": checks,
        "source_checks": source_checks,
        "access": {
            "synthetic_evaluator_and_gate_tests": 7,
            "confirmatory_records_evaluated": 0,
            "confirmatory_models_scored": 0,
            "development_models_rescored": 0,
        },
    }
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    if not audit["passed"]:
        print(json.dumps(audit, indent=2, sort_keys=True))
        raise SystemExit(1)
    lock = {
        "schema_version": "70-confirmatory-multi-environment",
        "experiment": "v70_confirmatory_evaluator_lock",
        "reporting_lock": str(reporting_path.relative_to(PROJECT_ROOT)),
        "reporting_lock_sha256": file_sha256(reporting_path),
        "family_implementation": str(family_path.relative_to(PROJECT_ROOT)),
        "family_implementation_sha256": file_sha256(family_path),
        "point_control_implementation": str(point_path.relative_to(PROJECT_ROOT)),
        "point_control_implementation_sha256": file_sha256(point_path),
        "unchanged_exact_record_evaluator": str(record_evaluator_path.relative_to(PROJECT_ROOT)),
        "unchanged_exact_record_evaluator_sha256": file_sha256(record_evaluator_path),
        "aggregation": str(aggregation_path.relative_to(PROJECT_ROOT)),
        "aggregation_sha256": file_sha256(aggregation_path),
        "evaluator": str(evaluator_path.relative_to(PROJECT_ROOT)),
        "evaluator_sha256": file_sha256(evaluator_path),
        "evaluator_tests": [str(path.relative_to(PROJECT_ROOT)) for path in tests],
        "evaluator_tests_sha256": [file_sha256(path) for path in tests],
        "evaluator_audit": str(audit_path.relative_to(PROJECT_ROOT)),
        "evaluator_audit_sha256": file_sha256(audit_path),
        "evaluator_auditor": str(auditor_path.relative_to(PROJECT_ROOT)),
        "evaluator_auditor_sha256": file_sha256(auditor_path),
        "attempt_path": "outputs/v70-confirmatory/evaluation/attempt.json",
        "expected_attempt_number": 1,
        "expected_records": census["record_count"],
        "expected_confirmatory_models": len(census["model_counts"]),
        "authorization": {
            "modify_prior_locks_code_census_reporting_or_gates": False,
            "run_confirmatory_outcome_once": True,
            "rescore_development_models": False,
            "drop_or_replace_models": False,
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
