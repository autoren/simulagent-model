#!/usr/bin/env python3
from __future__ import annotations

import argparse
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--result",
        default="outputs/v52-rao-blackwellized-particle-filtering/evaluation/result.json",
    )
    parser.add_argument(
        "--output",
        default="outputs/v52-rao-blackwellized-particle-filtering/post-result-audit.json",
    )
    parser.add_argument("--summary", default="docs/v52-results.md")
    args = parser.parse_args()
    result_path = (PROJECT_ROOT / args.result).resolve()
    output = (PROJECT_ROOT / args.output).resolve()
    summary = (PROJECT_ROOT / args.summary).resolve()
    result = json.loads(result_path.read_text())
    seal_path = PROJECT_ROOT / result["population_seal"]
    seal = json.loads(seal_path.read_text())
    implementation_path = PROJECT_ROOT / seal["implementation_lock"]
    implementation = json.loads(implementation_path.read_text())
    config = implementation["config_payload"]
    attempt_path = result_path.parent.parent / "evaluation-attempt.json"
    attempt = json.loads(attempt_path.read_text()) if attempt_path.is_file() else {}
    errors = []
    if result["population_seal_sha256"] != file_sha256(seal_path):
        errors.append("V52 result is not bound to sealed populations")
    if result.get("evaluation_run_number") != 1 or attempt.get("status") != "completed":
        errors.append("V52 one-run state is invalid")

    details = {}
    for name, artifact in result["detail_metrics"].items():
        path = PROJECT_ROOT / artifact["path"]
        if file_sha256(path) != artifact["sha256"]:
            errors.append(f"V52 {name} detail hash mismatch")
        details[name] = read(path)
    support_path = PROJECT_ROOT / result["audit_support"]["path"]
    if file_sha256(support_path) != result["audit_support"]["sha256"]:
        errors.append("V52 audit-support hash mismatch")
    support = json.loads(support_path.read_text())

    exact = aggregate_exact(
        details["exact"],
        support["controls"],
        support["exact_stream_collisions"],
        support["fingerprint_collisions"],
        support["fingerprint_pairs"],
        config,
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
        errors.append("V52 result does not reproduce from sealed detail metrics")

    audit = {
        "schema_version": 52,
        "experiment": "v52_post_result_audit",
        "passed": not errors,
        "decision": "accept_v52_result" if not errors else "reject_v52_integrity",
        "errors": errors,
        "result": str(result_path.relative_to(PROJECT_ROOT)),
        "result_sha256": file_sha256(result_path),
        "scientific_decision": result["decision"],
        "integrity_checks": {
            "seal_bound": result["population_seal_sha256"] == file_sha256(seal_path),
            "implementation_bound": seal["implementation_lock_sha256"]
            == file_sha256(implementation_path),
            "one_run": result.get("evaluation_run_number") == 1,
            "no_selection": result["data_access"]["selection_on_sealed_results"] == 0,
            "no_model": result["data_access"]["model_forward_passes"] == 0,
            "no_training": result["data_access"]["adapter_training_runs"] == 0,
            "non_final": not result["authorization"]["final_evaluation"],
            "active_selection_blocked": not result["authorization"]["active_intervention_selection"],
        },
        "reproduction_checks": reproduction,
    }
    output.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")

    metrics = result["metrics"]
    exact_metrics = metrics["exact"]
    primary = exact_metrics["by_budget"][str(exact_metrics["primary_budget"])]
    lines = [
        "# V52 results: Rao–Blackwellized particle filtering",
        "",
        f"Decision: `{result['decision']}`.",
        "",
        "V52 tests bounded particle inference only for the hidden dynamic world and delayed-effect queue. The 48 finite program/probability hypotheses and every particle's local one-step stochastic branches remain exactly enumerated. This is non-final and does not test continuous parameters, active intervention selection, reward, planning, language, noisy sensors, or open ontologies.",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Primary particle budget | {exact_metrics['primary_budget']} |",
        f"| Exact-benchmark completion | {exact_metrics['completed_fraction']:.6g} |",
        f"| Mean support-program TV | {primary['mean_support_program_tv']:.6g} |",
        f"| Mean query-program TV | {primary['mean_query_program_tv']:.6g} |",
        f"| Mean probability-marginal TV | {primary['mean_probability_marginal_tv']:.6g} |",
        f"| Mean joint-belief TV | {primary['mean_joint_belief_tv']:.6g} |",
        f"| Mean suffix-predictive TV | {primary['mean_suffix_predictive_tv']:.6g} |",
        f"| Mean absolute log-evidence error | {primary['mean_absolute_log_evidence_error']:.6g} |",
        f"| Minimum SBC chi-square p-value | {metrics['sbc']['minimum_chi_square_p_value']:.6g} |",
        f"| Maximum absolute rank-bin z | {metrics['sbc']['maximum_absolute_rank_bin_z']:.6g} |",
        f"| Maximum absolute coverage z | {metrics['sbc']['maximum_absolute_coverage_z']:.6g} |",
        f"| Unintended stream collisions | {exact_metrics['unintended_stream_collision_count']} |",
        f"| Independent-repeat fingerprint collision rate | {exact_metrics['stochastic_fingerprint_collision_rate']:.6g} |",
        f"| Scale-stress normalization rate | {metrics['scale']['normalization_rate']:.6g} |",
        "",
        "## Budget convergence",
        "",
        "| Particles | Mean core TV | Repeat dispersion | Mean log-evidence error |",
        "|---:|---:|---:|---:|",
    ]
    for budget, values in exact_metrics["by_budget"].items():
        lines.append(
            f"| {budget} | {values['mean_core_tv']:.6g} | "
            f"{values['mean_repeat_core_tv_dispersion']:.6g} | "
            f"{values['mean_absolute_log_evidence_error']:.6g} |"
        )
    lines.extend([
        "",
        f"All preregistered gates passed: `{str(result['qualification']['passed']).lower()}`.",
        f"Post-result integrity audit: `{'pass' if audit['passed'] else 'fail'}`.",
        "",
    ])
    summary.write_text("\n".join(lines))
    print(json.dumps(audit, indent=2, sort_keys=True))
    if not audit["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
