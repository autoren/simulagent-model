#!/usr/bin/env python3
"""Audit and freeze the V64 evaluator before its sole population access."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
from pathlib import Path
import subprocess
import sys

import numpy as np
import scipy

from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seal", default="configs/v64-population-seal.json")
    parser.add_argument(
        "--audit", default="outputs/v64-external-multi-action-eig/evaluator-audit.json"
    )
    parser.add_argument(
        "--output", default="configs/v64-evaluation-implementation-lock.json"
    )
    args = parser.parse_args()
    seal_path = (PROJECT_ROOT / args.seal).resolve()
    audit_path = (PROJECT_ROOT / args.audit).resolve()
    output_path = (PROJECT_ROOT / args.output).resolve()
    if output_path.exists():
        raise RuntimeError("V64 evaluator already frozen")
    seal = json.loads(seal_path.read_text())
    implementation_path = (PROJECT_ROOT / seal["implementation_lock"]).resolve()
    implementation = json.loads(implementation_path.read_text())
    errors: list[str] = []

    seal_ok = bool(
        seal["authorization"]["write_and_audit_evaluation_implementation"]
        and not seal["authorization"]["run_one_immutable_evaluation"]
        and file_sha256(implementation_path) == seal["implementation_lock_sha256"]
        and file_sha256(PROJECT_ROOT / seal["manifest"]) == seal["manifest_sha256"]
        and file_sha256(PROJECT_ROOT / seal["population_audit"])
        == seal["population_audit_sha256"]
        and file_sha256(PROJECT_ROOT / seal["generator"]) == seal["generator_sha256"]
        and file_sha256(PROJECT_ROOT / seal["seal_auditor"])
        == seal["seal_auditor_sha256"]
        and all(
            file_sha256(PROJECT_ROOT / row["path"]) == row["sha256"]
            for row in seal["files"].values()
        )
    )
    if not seal_ok:
        errors.append("V64 population seal or upstream implementation binding failed")

    for relative, digest in implementation["source_sha256"].items():
        if file_sha256(PROJECT_ROOT / relative) != digest:
            errors.append(f"frozen V64 exact source changed: {relative}")

    env = os.environ.copy()
    env["PYTHONPATH"] = str(PROJECT_ROOT / "python")
    tests = subprocess.run(
        [sys.executable, "-m", "unittest", "python/test_v64_evaluator.py"],
        cwd=PROJECT_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    tests_ok = tests.returncode == 0 and "Ran 5 tests" in tests.stderr and "OK" in tests.stderr
    if not tests_ok:
        errors.append("V64 evaluator synthetic tests failed")

    evaluator_path = PROJECT_ROOT / "python/evaluate_v64_eig.py"
    evaluator_source = evaluator_path.read_text()
    contracts = {
        "selection_uses_public_file_only": 'seal["files"]["selection_public"]' in evaluator_source
        and 'seal["files"]["selection_audit"]' not in evaluator_source,
        "all_four_actions_scored": "score_all_actions" in evaluator_source,
        "candidate_scalar_pairing_is_strict": "zip(candidate_scores, reference_scores, strict=True)" in evaluator_source,
        "selection_precedes_simulation": evaluator_source.index("select_action(family, belief)")
        < evaluator_source.index("simulate_step("),
        "outcome_leakage_control_present": "attempted_outcome_leak_selection" in evaluator_source,
        "paired_replication_interval": "normal_lower_95(difference)" in evaluator_source,
        "rank_support_uses_frozen_bins": 'spec["rankBins"]' in evaluator_source
        and 'spec["rankSupportSize"]' in evaluator_source,
        "noncompensatory_gate_conjunction": "passed = all(checks.values())" in evaluator_source,
        "one_shot_guard": "immutable evaluation result already exists" in evaluator_source,
        "approximate_and_reward_claims_false": '"approximate_particle_acquisition_tested": False' in evaluator_source
        and '"reward_planning_tested": False' in evaluator_source,
    }
    contracts_ok = all(contracts.values())
    if not contracts_ok:
        errors.append("V64 evaluator source violates a frozen contract")

    mutation_guards = {
        "omit_candidate_action": "candidate_action_comparisons" in evaluator_source
        and "candidate_omission_count" in evaluator_source,
        "unpaired_policy_interval": "np.asarray(results[\"adaptiveEIG\"][\"8\"]) - np.asarray(results[baseline][\"8\"])" in evaluator_source,
        "rank_tie_off_by_one": "equal + 1" in evaluator_source and "min(equal" in evaluator_source,
        "truth_field_to_selection": "truth_field_access_count" in evaluator_source
        and "assert_public_selection_payload" in evaluator_source,
        "post_outcome_selection": "realized_outcome_access_before_selection_count" in evaluator_source
        and "attempted_outcome_leak_selection" in evaluator_source,
        "compensatory_average_gate": "all(checks.values())" in evaluator_source,
        "fixed_random_label_swap": 'for baseline in ("fixed", "random")' in evaluator_source,
        "skip_SBC": "evaluate_sbc" in evaluator_source
        and "SBC_rank_chi_square" in evaluator_source,
    }
    mutations_ok = all(mutation_guards.values()) and len(mutation_guards) == 8
    if not mutations_ok:
        errors.append("V64 evaluator mutation guards are incomplete")

    result_path = PROJECT_ROOT / "outputs/v64-external-multi-action-eig/evaluation/result.json"
    downstream_absent = not result_path.exists() and not (
        PROJECT_ROOT / "configs/v64-outcome-lock.json"
    ).exists()
    if not downstream_absent:
        errors.append("V64 result or outcome lock exists before evaluator freeze")

    source_paths = [
        evaluator_path,
        PROJECT_ROOT / "python/test_v64_evaluator.py",
        Path(__file__).resolve(),
        PROJECT_ROOT / "python/v64_external_eig.py",
        PROJECT_ROOT / "python/v64_scalar_reference.py",
        PROJECT_ROOT / "python/v62_external_pomdp.py",
    ]
    audit = {
        "schema_version": 64,
        "experiment": "v64_evaluator_implementation_audit",
        "passed": not errors,
        "decision": "freeze_v64_evaluator_and_authorize_one_immutable_run" if not errors else "repair_v64_evaluator",
        "errors": errors,
        "checks": {
            "population_seal_and_upstream_bindings": seal_ok,
            "synthetic_evaluator_tests": tests_ok,
            "evaluator_contracts": contracts_ok,
            "mutation_guards": mutations_ok,
            "result_and_outcome_absent": downstream_absent,
        },
        "evaluator_contracts": contracts,
        "mutation_guards": mutation_guards,
        "unit_test_output": {"stdout": tests.stdout, "stderr": tests.stderr},
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
        },
        "source_sha256": {
            str(path.relative_to(PROJECT_ROOT)): file_sha256(path) for path in source_paths
        },
        "data_access": {
            "sealed_selection_records_accessed_by_candidate": 0,
            "sealed_adaptive_records_executed": 0,
            "sealed_SBC_records_executed": 0,
            "logical_evaluation_attempts": 0,
            "human_record_access_count": 0,
            "simulated_human_record_count": 0,
            "model_forward_pass_count": 0,
            "adapter_training_run_count": 0,
        },
    }
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    if errors:
        print(json.dumps(audit, indent=2, sort_keys=True))
        raise SystemExit(1)

    lock = {
        "schema_version": 64,
        "experiment": "v64_evaluation_implementation_lock",
        "population_seal": str(seal_path.relative_to(PROJECT_ROOT)),
        "population_seal_sha256": file_sha256(seal_path),
        "evaluator_audit": str(audit_path.relative_to(PROJECT_ROOT)),
        "evaluator_audit_sha256": file_sha256(audit_path),
        "source_sha256": audit["source_sha256"],
        "runtime": audit["runtime"],
        "operational_control_detection_rule": "mean_exact_EIG_regret_gt_0.001_nats_or_strict_selection_disagreement_gt_0.10_on_informative_records;_outcome_leakage_requires_firewall_rejection",
        "authorization": {
            "modify_v64_design_implementation_populations_or_evaluator": False,
            "run_one_immutable_v64_evaluation": True,
            "rerun_v64_evaluation": False,
            "approximate_particle_acquisition": False,
            "reward_planning": False,
            "formal_verification": False,
            "access_human_data": False,
            "simulate_human_data": False,
            "model_access": False,
            "adapter_training": False,
        },
    }
    lock["lock_payload_sha256"] = hashlib.sha256(
        json.dumps(lock, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    output_path.write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"audit": audit, "lock": lock}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
