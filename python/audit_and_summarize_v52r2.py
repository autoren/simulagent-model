#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json

from evaluate_v52_particle import (
    aggregate_exact,
    aggregate_scale,
    qualification,
    rank_diagnostics,
    read,
)
from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT


def maximum_numeric_delta(left, right):
    if isinstance(left, dict):
        if set(left) != set(right):
            return float("inf")
        return max((maximum_numeric_delta(left[key], right[key]) for key in left), default=0)
    if isinstance(left, list):
        if len(left) != len(right):
            return float("inf")
        return max((maximum_numeric_delta(a, b) for a, b in zip(left, right)), default=0)
    if isinstance(left, (int, float)) and not isinstance(left, bool):
        return abs(float(left) - float(right))
    return 0 if left == right else float("inf")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--result",
        default="outputs/v52r2-joint-normalization-repair/evaluation/result.json",
    )
    parser.add_argument(
        "--output",
        default="outputs/v52r2-joint-normalization-repair/post-result-audit.json",
    )
    parser.add_argument("--summary", default="docs/v52r2-results.md")
    args = parser.parse_args()
    result_path = (PROJECT_ROOT / args.result).resolve()
    output = (PROJECT_ROOT / args.output).resolve()
    summary = (PROJECT_ROOT / args.summary).resolve()
    result = json.loads(result_path.read_text())
    repair_implementation_path = PROJECT_ROOT / result["repair_implementation_lock"]
    repair_implementation = json.loads(repair_implementation_path.read_text())
    repair_lock_path = PROJECT_ROOT / repair_implementation["repair_lock"]
    repair_lock = json.loads(repair_lock_path.read_text())
    source_outcome = json.loads(
        (PROJECT_ROOT / repair_lock["source_outcome_lock"]).read_text()
    )
    source_result = json.loads((PROJECT_ROOT / source_outcome["result"]).read_text())
    seal_path = PROJECT_ROOT / result["population_seal"]
    seal = json.loads(seal_path.read_text())
    source_implementation = json.loads(
        (PROJECT_ROOT / seal["implementation_lock"]).read_text()
    )
    config = source_implementation["config_payload"]
    attempt_path = result_path.parent.parent / "evaluation-attempt.json"
    attempt = json.loads(attempt_path.read_text()) if attempt_path.is_file() else {}
    errors = []
    chain_bound = (
        result["repair_implementation_lock_sha256"]
        == file_sha256(repair_implementation_path)
        and repair_implementation["repair_lock_sha256"] == file_sha256(repair_lock_path)
        and result["population_seal_sha256"] == file_sha256(seal_path)
        and repair_lock["source_population_seal_sha256"] == file_sha256(seal_path)
    )
    if not chain_bound:
        errors.append("V52r2 result is not bound to the repair and source seal chain")
    if (
        result.get("repair_evaluation_run_number") != 1
        or result.get("evaluation_run_number") != 1
        or attempt.get("status") != "completed"
    ):
        errors.append("V52r2 one-repair-run state is invalid")

    details = {}
    for name, artifact in result["detail_metrics"].items():
        path = PROJECT_ROOT / artifact["path"]
        if file_sha256(path) != artifact["sha256"]:
            errors.append(f"V52r2 {name} detail hash mismatch")
        details[name] = read(path)
    support_path = PROJECT_ROOT / result["audit_support"]["path"]
    if file_sha256(support_path) != result["audit_support"]["sha256"]:
        errors.append("V52r2 audit-support hash mismatch")
    support = json.loads(support_path.read_text())
    exact = aggregate_exact(
        details["exact"], support["controls"],
        support["exact_stream_collisions"], support["fingerprint_collisions"],
        support["fingerprint_pairs"], config,
    )
    exact["unintended_stream_collision_count"] += (
        support["sbc_stream_collisions"] + support["scale_stream_collisions"]
    )
    reproduced_metrics = {
        "exact": exact,
        "sbc": rank_diagnostics(details["sbc"], config),
        "sbc_normalization_rate": sum(
            row["normalization"] for row in details["sbc"]
        ) / len(details["sbc"]),
        "sbc_target_program_extinction_rate": sum(
            row["target_program_extinct"] for row in details["sbc"]
        ) / len(details["sbc"]),
        "scale": aggregate_scale(details["scale"]),
    }
    reproduced_qualification = qualification(reproduced_metrics, config["gates"])
    reproduction = {
        "metrics": reproduced_metrics == result["metrics"],
        "qualification": reproduced_qualification == result["qualification"],
    }
    if not all(reproduction.values()):
        errors.append("V52r2 metrics or qualification do not reproduce")

    exact_delta = maximum_numeric_delta(
        source_result["metrics"]["exact"], result["metrics"]["exact"]
    )
    sbc_unchanged = source_result["metrics"]["sbc"] == result["metrics"]["sbc"]
    scale_source = copy.deepcopy(source_result["metrics"]["scale"])
    scale_repair = copy.deepcopy(result["metrics"]["scale"])
    scale_source.pop("normalization_rate")
    scale_repair.pop("normalization_rate")
    scale_unchanged = scale_source == scale_repair
    ancillary_unchanged = (
        source_result["metrics"]["sbc_target_program_extinction_rate"]
        == result["metrics"]["sbc_target_program_extinction_rate"]
    )
    normalization_repaired = (
        result["metrics"]["sbc_normalization_rate"] == 1.0
        and result["metrics"]["scale"]["normalization_rate"] == 1.0
    )
    substantive_invariance = (
        exact_delta <= float(repair_lock["maximum_permitted_base_vs_repair_tv"])
        and sbc_unchanged and scale_unchanged and ancillary_unchanged
    )
    if not substantive_invariance:
        errors.append("V52r2 changed a substantive metric beyond its repair boundary")
    if not normalization_repaired:
        errors.append("V52r2 did not repair both normalization rates")
    if not result["qualification"]["passed"]:
        errors.append("V52r2 does not pass the unchanged noncompensatory gates")

    audit = {
        "schema_version": 52,
        "revision": "r2",
        "experiment": "v52r2_post_result_audit",
        "passed": not errors,
        "decision": "accept_v52r2_repair" if not errors else "reject_v52r2_integrity",
        "errors": errors,
        "result": str(result_path.relative_to(PROJECT_ROOT)),
        "result_sha256": file_sha256(result_path),
        "scientific_decision": result["decision"],
        "integrity_checks": {
            "repair_and_source_chain_bound": chain_bound,
            "one_repair_run": result.get("repair_evaluation_run_number") == 1,
            "total_evaluation_runs_disclosed": result["data_access"]["particle_evaluation_runs"] == 2,
            "repair_selection_disclosed": result["data_access"]["repair_selected_from_frozen_source_outcome"],
            "no_model": result["data_access"]["model_forward_passes"] == 0,
            "no_training": result["data_access"]["adapter_training_runs"] == 0,
            "active_selection_blocked": not result["authorization"]["active_intervention_selection"],
            "non_final": not result["authorization"]["final_evaluation"],
        },
        "reproduction_checks": reproduction,
        "repair_boundary_checks": {
            "maximum_source_vs_repair_exact_metric_delta": exact_delta,
            "sbc_diagnostics_unchanged": sbc_unchanged,
            "scale_metrics_except_normalization_unchanged": scale_unchanged,
            "ancillary_metrics_unchanged": ancillary_unchanged,
            "both_normalization_rates_one": normalization_repaired,
            "all_unchanged_gates_pass": result["qualification"]["passed"],
        },
    }
    output.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")

    metrics = result["metrics"]
    exact_metrics = metrics["exact"]
    primary = exact_metrics["by_budget"][str(exact_metrics["primary_budget"])]
    lines = [
        "# V52r2 results: final-joint Decimal normalization repair",
        "",
        f"Decision: `{result['decision']}`.",
        "",
        "V52r2 reruns the unchanged sealed V52 populations with only final joint/configuration assembly moved into the preregistered 100-digit Decimal context. Particle filtering, likelihoods, resampling paths, budgets, seeds, exact oracle, metrics, and gates are unchanged.",
        "",
        "| Metric | V52r2 |",
        "|---|---:|",
        f"| SBC normalization rate | {metrics['sbc_normalization_rate']:.6g} |",
        f"| Scale normalization rate | {metrics['scale']['normalization_rate']:.6g} |",
        f"| Primary mean joint-belief TV | {primary['mean_joint_belief_tv']:.6g} |",
        f"| Primary mean query-program TV | {primary['mean_query_program_tv']:.6g} |",
        f"| Primary mean log-evidence error | {primary['mean_absolute_log_evidence_error']:.6g} |",
        f"| Minimum SBC chi-square p-value | {metrics['sbc']['minimum_chi_square_p_value']:.6g} |",
        f"| Maximum source-vs-repair exact-metric delta | {exact_delta:.6g} |",
        f"| Unintended stream collisions | {exact_metrics['unintended_stream_collision_count']} |",
        "",
        f"All unchanged preregistered gates passed: `{str(result['qualification']['passed']).lower()}`.",
        f"Post-result repair-boundary audit: `{'pass' if audit['passed'] else 'fail'}`.",
        "",
    ]
    summary.write_text("\n".join(lines))
    print(json.dumps(audit, indent=2, sort_keys=True))
    if not audit["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
