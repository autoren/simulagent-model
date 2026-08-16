#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json

from evaluate_v53_smc2 import (
    aggregate_exact,
    aggregate_scale,
    qualification,
    rank_diagnostics,
)
from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT


def read(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--result", default="outputs/v53r2-continuous-parameter-smc2/evaluation/result.json"
    )
    parser.add_argument(
        "--output", default="outputs/v53r2-continuous-parameter-smc2/post-result-audit.json"
    )
    parser.add_argument("--summary", default="docs/v53r2-results.md")
    args = parser.parse_args()
    result_path, output, summary = tuple(
        (PROJECT_ROOT / value).resolve()
        for value in (args.result, args.output, args.summary)
    )
    result = json.loads(result_path.read_text())
    seal_path = PROJECT_ROOT / result["population_seal"]
    seal = json.loads(seal_path.read_text())
    implementation_path = PROJECT_ROOT / seal["implementation_lock"]
    implementation = json.loads(implementation_path.read_text())
    config = implementation["config_payload"]
    errors = []
    implementation_ok = all(
        file_sha256(PROJECT_ROOT / path) == expected
        for path, expected in implementation["implementation"].items()
    )
    seal_ok = (
        result["population_seal_sha256"] == file_sha256(seal_path)
        and seal["implementation_lock_sha256"] == file_sha256(implementation_path)
    )
    details = {}
    detail_hashes_ok = True
    for name, artifact in result["detail_metrics"].items():
        path = PROJECT_ROOT / artifact["path"]
        detail_hashes_ok &= file_sha256(path) == artifact["sha256"]
        details[name] = read(path)
    attempt = result_path.parent.parent / "evaluation-attempt.json"
    attempt_value = json.loads(attempt.read_text())
    one_run_ok = (
        result["evaluation_run_number"] == 1
        and attempt_value["status"] == "completed"
        and attempt_value["evaluation_run"] == 1
        and attempt_value["result_sha256"] == file_sha256(result_path)
        and result["data_access"]["smc_squared_evaluation_runs"] == 1
        and result["data_access"]["pmcmc_reference_runs"] == 1
        and result["data_access"]["selection_on_sealed_results"] == 0
    )
    if not implementation_ok:
        errors.append("implementation hashes changed after lock")
    if not seal_ok or not detail_hashes_ok:
        errors.append("population seal or detail metric hashes are invalid")
    if not one_run_ok:
        errors.append("single-run firewall or attempt binding failed")

    reproduced_exact = aggregate_exact(
        details["exact"], details["pmcmc"], details["controls"],
        {"outer": [], "inner": []}, {"outer": [], "inner": []}, config,
    )
    exact = result["metrics"]["exact"]
    exact_reproduction_ok = all((
        reproduced_exact["completed_fraction"] == exact["completed_fraction"],
        reproduced_exact["primary_budget"] == exact["primary_budget"],
        reproduced_exact["by_budget"] == exact["by_budget"],
        reproduced_exact["controls"] == exact["controls"],
        reproduced_exact["pmcmc"] == exact["pmcmc"],
        reproduced_exact["primary_minus_medium_mean_error"]
        == exact["primary_minus_medium_mean_error"],
        reproduced_exact["medium_minus_low_mean_error"]
        == exact["medium_minus_low_mean_error"],
    ))
    sbc_reproduction_ok = rank_diagnostics(details["sbc"], config) == result["metrics"]["sbc"]
    scale_reproduction_ok = aggregate_scale(details["scale"]) == result["metrics"]["scale"]
    qualification_reproduction = qualification(result["metrics"], config["gates"])
    qualification_ok = qualification_reproduction == result["qualification"]
    if not all((exact_reproduction_ok, sbc_reproduction_ok, scale_reproduction_ok, qualification_ok)):
        errors.append("metric or qualification reproduction failed")

    audit = {
        "schema_version": 53,
        "revision": "r2",
        "experiment": "v53r2_post_result_audit",
        "passed": not errors,
        "errors": errors,
        "result": str(result_path.relative_to(PROJECT_ROOT)),
        "result_sha256": file_sha256(result_path),
        "population_seal": str(seal_path.relative_to(PROJECT_ROOT)),
        "population_seal_sha256": file_sha256(seal_path),
        "checks": {
            "implementation_integrity": implementation_ok,
            "population_seal_binding": seal_ok,
            "detail_hashes": detail_hashes_ok,
            "single_run_firewall": one_run_ok,
            "exact_metric_reproduction": exact_reproduction_ok,
            "sbc_metric_reproduction": sbc_reproduction_ok,
            "scale_metric_reproduction": scale_reproduction_ok,
            "qualification_reproduction": qualification_ok,
        },
        "qualification_passed": result["qualification"]["passed"],
        "decision": result["decision"],
        "data_access": result["data_access"],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    primary = exact["by_budget"][str(exact["primary_budget"])]
    failed = [
        name for name, passed in result["qualification"]["checks"].items() if not passed
    ]
    summary.write_text(
        "# V53r2 continuous-parameter inference results\n\n"
        f"**Qualification:** {'PASS' if result['qualification']['passed'] else 'FAIL'}  \n"
        f"**Decision:** `{result['decision']}`\n\n"
        "## Primary exact-oracle agreement\n\n"
        f"- Program TV: mean `{primary['mean_program_tv']:.6g}`, q95 `{primary['q95_program_tv']:.6g}`\n"
        f"- Theta Wasserstein: mean `{primary['mean_theta_wasserstein']:.6g}`, q95 `{primary['q95_theta_wasserstein']:.6g}`\n"
        f"- Binned joint TV: mean `{primary['mean_binned_program_theta_tv']:.6g}`, q95 `{primary['q95_binned_program_theta_tv']:.6g}`\n"
        f"- Configuration TV: mean `{primary['mean_configuration_tv']:.6g}`, q95 `{primary['q95_configuration_tv']:.6g}`\n"
        f"- Suffix TV: mean `{primary['mean_suffix_predictive_tv']:.6g}`, q95 `{primary['q95_suffix_predictive_tv']:.6g}`\n"
        f"- Mean absolute log-evidence error: `{primary['mean_absolute_log_evidence_error']:.6g}`\n\n"
        "## Calibration and PMCMC\n\n"
        f"- SBC minimum chi-square p: `{result['metrics']['sbc']['minimum_chi_square_p_value']:.6g}`\n"
        f"- SBC maximum rank-bin z: `{result['metrics']['sbc']['maximum_absolute_rank_bin_z']:.6g}`\n"
        f"- SBC maximum coverage z: `{result['metrics']['sbc']['maximum_absolute_coverage_z']:.6g}`\n"
        f"- PMCMC mean acceptance: `{exact['pmcmc']['mean_acceptance_rate']:.6g}`\n"
        f"- PMCMC maximum split-Rhat: `{exact['pmcmc']['maximum_split_rhat']:.6g}`\n"
        f"- PMCMC minimum bulk ESS: `{exact['pmcmc']['minimum_bulk_ess']:.6g}`\n"
        f"- PMCMC maximum theta Wasserstein: `{exact['pmcmc']['maximum_theta_wasserstein']:.6g}`\n\n"
        "## Gate status\n\n"
        f"Failed gates: `{failed}`\n"
    )
    print(json.dumps(audit, indent=2, sort_keys=True))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
