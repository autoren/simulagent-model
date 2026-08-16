#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json

from evaluate_v51_sbc import aggregate, qualification, read
from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--result",
        default="outputs/v51-simulation-based-calibration/calibration/result.json",
    )
    parser.add_argument(
        "--output",
        default="outputs/v51-simulation-based-calibration/post-result-audit.json",
    )
    parser.add_argument("--summary", default="docs/v51-results.md")
    args = parser.parse_args()
    result_path = (PROJECT_ROOT / args.result).resolve()
    output = (PROJECT_ROOT / args.output).resolve()
    summary = (PROJECT_ROOT / args.summary).resolve()
    result = json.loads(result_path.read_text())
    seal_path = PROJECT_ROOT / result["corpus_seal"]
    seal = json.loads(seal_path.read_text())
    implementation_path = PROJECT_ROOT / seal["implementation_lock"]
    implementation = json.loads(implementation_path.read_text())
    attempt_path = result_path.parent.parent / "calibration-attempt.json"
    attempt = json.loads(attempt_path.read_text()) if attempt_path.is_file() else {}
    errors = []
    if result["corpus_seal_sha256"] != file_sha256(seal_path):
        errors.append("V51 result is not bound to sealed corpus")
    if result.get("calibration_run_number") != 1 or attempt.get("status") != "completed":
        errors.append("V51 one-run state is invalid")
    details_path = PROJECT_ROOT / result["replication_metrics"]
    if file_sha256(details_path) != result["replication_metrics_sha256"]:
        errors.append("V51 replication metrics hash mismatch")
    details = read(details_path)
    reproduced_metrics = aggregate(details, implementation["config_payload"])
    reproduced_qualification = qualification(
        reproduced_metrics, implementation["config_payload"]["gates"]
    )
    reproduction = {
        "metrics": reproduced_metrics == result["metrics"],
        "qualification": reproduced_qualification == result["qualification"],
    }
    if not all(reproduction.values()):
        errors.append("V51 result does not reproduce from sealed replication metrics")
    audit = {
        "schema_version": 51,
        "experiment": "v51_post_result_audit",
        "passed": not errors,
        "decision": "accept_v51_calibration" if not errors else "reject_v51_integrity",
        "errors": errors,
        "result": str(result_path.relative_to(PROJECT_ROOT)),
        "result_sha256": file_sha256(result_path),
        "scientific_decision": result["decision"],
        "integrity_checks": {
            "seal_bound": result["corpus_seal_sha256"] == file_sha256(seal_path),
            "implementation_bound": seal["implementation_lock_sha256"]
            == file_sha256(implementation_path),
            "one_run": result.get("calibration_run_number") == 1,
            "no_selection": result["data_access"]["selection_on_calibration_replications"] == 0,
            "no_model": result["data_access"]["model_forward_passes"] == 0,
            "no_training": result["data_access"]["adapter_training_runs"] == 0,
            "non_final": not result["authorization"]["final_evaluation"],
        },
        "reproduction_checks": reproduction,
    }
    output.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")

    metrics = result["metrics"]
    primary = metrics["primary_sbc"]
    lines = [
        "# V51 results: simulation-based calibration of exact joint inference",
        "",
        f"Decision: `{result['decision']}`.",
        "",
        "V51 is a non-final, prior-predictive calibration test of exact discrete inference over the mechanic program and the current world/queue configuration. It also compares two independently implemented exact paths. It does not test particle approximation, intervention selection, reward, planning, language, noisy sensors, continuous parameters, or open ontologies.",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Completed replications | {metrics['replications']} |",
        f"| Normalization rate | {metrics['normalization_rate']:.6g} |",
        f"| Maximum exact-path TV | {metrics['maximum_exact_path_tv']:.6g} |",
        f"| Minimum primary rank chi-square p-value | {primary['minimum_chi_square_p_value']:.6g} |",
        f"| Maximum absolute primary rank-bin z | {primary['maximum_absolute_rank_bin_z']:.6g} |",
        f"| Maximum absolute primary coverage z | {primary['maximum_absolute_coverage_z']:.6g} |",
        f"| Bug controls rejected | {metrics['bug_controls_rejected']} / {len(metrics['bug_controls'])} |",
        "",
        "## Bug-sensitivity controls",
        "",
        "| Control | Rejected | Minimum p | Max rank z | Max coverage z |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, control in metrics["bug_controls"].items():
        lines.append(
            f"| {name} | {str(control['rejected']).lower()} | "
            f"{control['minimum_chi_square_p_value']:.6g} | "
            f"{control['maximum_absolute_rank_bin_z']:.6g} | "
            f"{control['maximum_absolute_coverage_z']:.6g} |"
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
