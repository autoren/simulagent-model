#!/usr/bin/env python3
"""Audit V62r1 rescore aggregation before its single authorized execution."""
from __future__ import annotations

import argparse
import ast
import json

from evaluate_v62r1_repair import qualify
from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--implementation-lock", default="configs/v62r1-implementation-lock.json"
    )
    parser.add_argument("--evaluator", default="python/evaluate_v62r1_repair.py")
    parser.add_argument(
        "--post-audit", default="python/audit_and_summarize_v62r1.py"
    )
    parser.add_argument("--freezer", default="python/freeze_v62r1_outcome.py")
    parser.add_argument(
        "--output",
        default="outputs/v62r1-terminal-residual-repair/evaluation-implementation-audit.json",
    )
    args = parser.parse_args()
    implementation_lock_path, evaluator_path, post_audit_path, freezer_path, output = tuple(
        (PROJECT_ROOT / value).resolve()
        for value in (
            args.implementation_lock,
            args.evaluator,
            args.post_audit,
            args.freezer,
            args.output,
        )
    )
    implementation_lock = json.loads(implementation_lock_path.read_text())
    design = json.loads((PROJECT_ROOT / implementation_lock["design_lock"]).read_text())
    config = design["config_payload"]
    errors: list[str] = []

    source_ok = (
        implementation_lock["authorization"]["run_one_repair_rescore"]
        and file_sha256(PROJECT_ROOT / implementation_lock["implementation"])
        == implementation_lock["implementation_sha256"]
        and file_sha256(PROJECT_ROOT / implementation_lock["source_v62_outcome_lock"])
        == implementation_lock["source_v62_outcome_lock_sha256"]
        and file_sha256(PROJECT_ROOT / implementation_lock["source_v62_external_bundle_seal"])
        == implementation_lock["source_v62_external_bundle_seal_sha256"]
    )
    if not source_ok:
        errors.append("V62r1 implementation or immutable source binding changed")

    evaluator_text = evaluator_path.read_text()
    tree = ast.parse(evaluator_text)
    imported_names = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }
    offline_rescore = (
        "subprocess" not in imported_names
        and "requests" not in imported_names
        and "random" not in imported_names
        and "terminal_aware_bellman_residual" in evaluator_text
        and "official_v62_rollout" not in evaluator_text
        and "new_external_rollouts" in evaluator_text
    )
    if not offline_rescore:
        errors.append("V62r1 evaluator is not an offline immutable-node rescore")

    gates = config["gates"]
    passing_metrics = {
        "source_binding_rate": 1.0,
        "source_old_residual_reproduction_rate": 1.0,
        "old_failure_terminal_localization_rate": 1.0,
        "maximum_corrected_residual": 0.0,
        "maximum_corrected_terminal_residual": 0.0,
        "maximum_corrected_nonterminal_residual": 0.0,
        "other_v62_gate_reproduction_rate": 1.0,
        "maximum_exact_record_delta": 0.0,
        "maximum_official_rollout_record_delta": 0.0,
        "repair_fixture_pass_rate": 1.0,
        "repair_mutant_kill_rate": 1.0,
        "repair_attempt_and_access_violation_count": 0,
    }
    passing = qualify(passing_metrics, gates)
    failing_values = {
        "source_binding_rate": 0.0,
        "source_old_residual_reproduction_rate": 0.0,
        "old_failure_terminal_localization_rate": 0.0,
        "maximum_corrected_residual": 1.0,
        "maximum_corrected_terminal_residual": 1.0,
        "maximum_corrected_nonterminal_residual": 1.0,
        "other_v62_gate_reproduction_rate": 0.0,
        "maximum_exact_record_delta": 1.0,
        "maximum_official_rollout_record_delta": 1.0,
        "repair_fixture_pass_rate": 0.0,
        "repair_mutant_kill_rate": 0.0,
        "repair_attempt_and_access_violation_count": 1,
    }
    gate_mutants_killed = {}
    for metric, bad_value in failing_values.items():
        mutant = dict(passing_metrics)
        mutant[metric] = bad_value
        judged = qualify(mutant, gates)
        gate_mutants_killed[metric] = (
            not judged["passed"]
            and sum(not value for value in judged["checks"].values()) == 1
        )
    aggregation_ok = (
        passing["passed"]
        and len(passing["checks"]) == len(gates) == 12
        and all(gate_mutants_killed.values())
    )
    if not aggregation_ok:
        errors.append("not all 12 noncompensatory V62r1 gates are enforced")

    post_source = post_audit_path.read_text()
    freezer_source = freezer_path.read_text()
    reporting_ok = (
        "original_v62_remains_failed" in post_source
        and "original_v62_qualification_passed" in freezer_source
        and "treat_original_v62_as_passing" in freezer_source
        and "measurement_repair_not_independent_replication" in freezer_source
    )
    if not reporting_ok:
        errors.append("V62r1 reporting could relabel the original V62 failure")

    downstream_absent = not any(
        (PROJECT_ROOT / path).exists()
        for path in (
            "configs/v62r1-evaluation-implementation-lock.json",
            "configs/v62r1-outcome-lock.json",
            "outputs/v62r1-terminal-residual-repair/rescore-attempt.json",
            "outputs/v62r1-terminal-residual-repair/rescore/result.json",
            "outputs/v62r1-terminal-residual-repair/post-result-audit.json",
            "docs/v62r1-results.md",
        )
    )
    if not downstream_absent:
        errors.append("V62r1 result exists before evaluation implementation lock")

    result = {
        "schema_version": "62r1",
        "experiment": "v62r1_evaluation_implementation_audit",
        "passed": not errors,
        "decision": (
            "freeze_v62r1_evaluation_implementation"
            if not errors
            else "repair_v62r1_evaluation_implementation"
        ),
        "errors": errors,
        "implementation_lock": str(implementation_lock_path.relative_to(PROJECT_ROOT)),
        "implementation_lock_sha256": file_sha256(implementation_lock_path),
        "evaluation_files_sha256": {
            str(path.relative_to(PROJECT_ROOT)): file_sha256(path)
            for path in (evaluator_path, post_audit_path, freezer_path)
        },
        "gate_mutants_killed": gate_mutants_killed,
        "checks": {
            "frozen_repair_and_source_bindings": source_ok,
            "offline_immutable_node_rescore": offline_rescore,
            "all_twelve_noncompensatory_gates": aggregation_ok,
            "failure_preserving_reporting": reporting_ok,
            "downstream_absence": downstream_absent,
        },
        "data_access": {
            "repair_rescores": 0,
            "new_candidate_evaluations": 0,
            "new_external_rollouts": 0,
            "human_records": 0,
            "model_forward_passes": 0,
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
