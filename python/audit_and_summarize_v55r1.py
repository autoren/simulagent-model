#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from evaluate_v55r1_planning import aggregate
from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--result",
        default="outputs/v55r1-delayed-consequence-adequacy-confirmation/evaluation/result.json",
    )
    parser.add_argument(
        "--audit",
        default="outputs/v55r1-delayed-consequence-adequacy-confirmation/post-result-audit.json",
    )
    parser.add_argument("--summary", default="docs/v55r1-results.md")
    args = parser.parse_args()
    result_path, audit_path, summary_path = tuple(
        (PROJECT_ROOT / value).resolve()
        for value in (args.result, args.audit, args.summary)
    )
    if audit_path.exists() or summary_path.exists():
        raise FileExistsError("V55r1 post-result artifacts already exist")
    result = json.loads(result_path.read_text())
    seal_path = PROJECT_ROOT / result["population_seal"]
    seal = json.loads(seal_path.read_text())
    implementation = json.loads(
        (PROJECT_ROOT / seal["implementation_lock"]).read_text()
    )
    design = json.loads(
        (PROJECT_ROOT / implementation["design_lock"]).read_text()
    )
    config = design["config_payload"]
    errors: list[str] = []

    sealed = (
        result["population_seal_sha256"] == file_sha256(seal_path)
        and seal["population_sha256"]
        == file_sha256(PROJECT_ROOT / seal["population"])
        and result["evaluation_run"] == 1
        and result["evaluation_implementation_lock_sha256"]
        == file_sha256(PROJECT_ROOT / result["evaluation_implementation_lock"])
    )
    if not sealed:
        errors.append("V55r1 result is not bound to the sealed inputs")

    records = result["records"]
    records_ok = (
        len(records) == 16
        and [row["record"] for row in records] == list(range(16))
        and len({row["id"] for row in records}) == 16
        and all(set((
            "id", "record", "history_class", "goal", "belief_atoms",
            "root_value", "reference_value", "root_value_error",
            "root_optimal_set_member", "independent_policy_value",
            "independent_policy_error", "selected_action_key",
            "delay_suppressed_value", "delay_suppressed_optimal_action_keys",
            "root_action_changes_under_delay_suppression",
            "absolute_root_value_change_under_delay_suppression",
            "delayed_consequence_sensitive", "integrity", "finite",
        )) <= set(row) for row in records)
    )
    if not records_ok:
        errors.append("V55r1 result records are incomplete, duplicated, or reordered")

    reproduced = aggregate(records, config)
    metrics_ok = reproduced["metrics"] == result["metrics"]
    checks_ok = reproduced["checks"] == result["qualification"]["checks"]
    qualification_ok = reproduced["passed"] == result["qualification"]["passed"]
    if not metrics_ok:
        errors.append("V55r1 aggregate metrics do not reproduce")
    if not checks_ok or not qualification_ok:
        errors.append("V55r1 qualification does not reproduce")

    exact_references_ok = all(
        row["root_value_error"] <= 1e-10
        and row["independent_policy_error"] <= 1e-10
        and row["root_optimal_set_member"]
        for row in records
    )
    if not exact_references_ok:
        errors.append("V55r1 independent exact references disagree")

    integrity = result["metrics"]["integrity"]
    integrity_ok = all(value == 0 for value in integrity.values())
    if not integrity_ok:
        errors.append("V55r1 selection, action, or stream integrity failed")

    audit = {
        "schema_version": 55,
        "revision": "r1",
        "experiment": "v55r1_post_result_audit",
        "passed": not errors,
        "decision": "accept_v55r1_result" if not errors else "invalidate_v55r1_result",
        "errors": errors,
        "result": str(result_path.relative_to(PROJECT_ROOT)),
        "result_sha256": file_sha256(result_path),
        "population_seal": str(seal_path.relative_to(PROJECT_ROOT)),
        "population_seal_sha256": file_sha256(seal_path),
        "qualification": result["qualification"],
        "checks": {
            "sealed_bindings": sealed,
            "record_count_ids_and_schema": records_ok,
            "metric_aggregation_reproduced": metrics_ok,
            "qualification_reproduced": checks_ok and qualification_ok,
            "independent_exact_references": exact_references_ok,
            "selection_action_and_stream_integrity": integrity_ok,
        },
        "data_access": {
            "additional_v55r1_planning_evaluation_runs": 0,
            "additional_v55_planning_evaluation_runs": 0,
            "model_forward_passes": 0,
            "adapter_training_runs": 0,
        },
    }
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")

    delayed = result["metrics"]["delayed_consequence"]
    exact = result["metrics"]["exact_correctness"]
    if result["qualification"]["passed"]:
        decision = (
            "V55 remains a frozen 19/20 failure; V55r1 independently confirms "
            "the missing delayed-consequence capability. Together they authorize "
            "only preregistration of symbolic and probabilistic policy verification."
        )
    else:
        decision = (
            "V55 remains failed and V55r1 does not repair the missing delayed-"
            "consequence evidence. Formal-verification preregistration remains blocked."
        )
    summary_path.write_text(
        "# V55r1 results: delayed-consequence adequacy confirmation\n\n"
        f"Decision: {decision}\n\n"
        "V55r1 is a supplementary confirmation only; it never relabels V55 as "
        "a standalone pass.\n\n"
        "## Sealed results\n\n"
        f"- All V55r1 qualification gates passed: `{result['qualification']['passed']}`.\n"
        f"- Delayed-sensitive policy fraction: `{delayed['delayed_consequence_sensitive_policy_fraction']}`.\n"
        f"- Delayed-sensitive tasks: `{delayed['delayed_consequence_sensitive_task_count']}/16`.\n"
        f"- By history class: `{json.dumps(delayed['delayed_sensitive_task_count_by_history_class'], sort_keys=True)}`.\n"
        f"- Root-action change fraction: `{delayed['root_action_change_fraction_under_delay_suppression']}`.\n"
        f"- Root-value change fraction over 0.001: `{delayed['root_value_change_fraction_over_0_001']}`.\n"
        f"- Mean absolute root-value change: `{delayed['mean_absolute_root_value_change_under_delay_suppression']}`.\n"
        f"- Maximum root/reference error: `{exact['maximum_root_value_error_against_scalar_reference']}`.\n"
        f"- Maximum independent policy-evaluation error: `{exact['maximum_independent_policy_evaluation_error']}`.\n"
        f"- Integrity violations: `{sum(integrity.values())}`.\n\n"
        "## Boundary\n\n"
        "A full pass may be combined only with V55's nineteen frozen passing gates. "
        "It authorizes a new verification preregistration, not execution of formal "
        "verification, longer-horizon claims, approximate search, language grounding, "
        "model access, or training.\n"
    )
    print(json.dumps(audit, indent=2, sort_keys=True))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
