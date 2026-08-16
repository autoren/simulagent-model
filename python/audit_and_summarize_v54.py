#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json

from evaluate_v54_eig import (
    aggregate_selection,
    qualification,
    rank_diagnostics,
    read_jsonl,
)
from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", default="outputs/v54-exact-one-step-eig/evaluation/result.json")
    parser.add_argument("--output", default="outputs/v54-exact-one-step-eig/post-result-audit.json")
    parser.add_argument("--summary", default="docs/v54-results.md")
    args = parser.parse_args()
    result_path, output, summary = tuple(
        (PROJECT_ROOT / value).resolve()
        for value in (args.result, args.output, args.summary)
    )
    result = json.loads(result_path.read_text())
    design = json.loads((PROJECT_ROOT / "configs/v54-design-lock.json").read_text())
    config = design["config_payload"]
    errors = []
    detail_hashes = all(
        file_sha256(PROJECT_ROOT / artifact["path"]) == artifact["sha256"]
        for artifact in result["detail_metrics"].values()
    )
    if not detail_hashes:
        errors.append("V54 detail metric hash mismatch")
    selection_rows = read_jsonl(PROJECT_ROOT / result["detail_metrics"]["selection"]["path"])
    adaptive_rows = read_jsonl(PROJECT_ROOT / result["detail_metrics"]["adaptive_sbc"]["path"])
    reproduced_selection = aggregate_selection(selection_rows, config)
    reproduced_sbc = rank_diagnostics(adaptive_rows, config)
    reproduction = (
        reproduced_selection == result["metrics"]["selection"]
        and reproduced_sbc == result["metrics"]["adaptive_sbc"]
    )
    if not reproduction:
        errors.append("V54 aggregate metrics do not reproduce from detail rows")
    reproduced_qualification = qualification(result["metrics"], config["gates"])
    qualification_ok = reproduced_qualification == result["qualification"]
    if not qualification_ok:
        errors.append("V54 qualification does not reproduce")
    attempt = json.loads(
        (PROJECT_ROOT / "outputs/v54-exact-one-step-eig/evaluation-attempt.json").read_text()
    )
    single_run = (
        attempt["status"] == "completed"
        and attempt["evaluation_run"] == result["evaluation_run_number"] == 1
        and attempt["result_sha256"] == file_sha256(result_path)
    )
    if not single_run:
        errors.append("V54 single-run firewall did not bind the result")
    audit = {
        "schema_version": 54,
        "experiment": "v54_post_result_audit",
        "passed": not errors,
        "qualification_passed": result["qualification"]["passed"],
        "errors": errors,
        "result": str(result_path.relative_to(PROJECT_ROOT)),
        "result_sha256": file_sha256(result_path),
        "checks": {
            "detail_hashes": detail_hashes,
            "aggregate_metric_reproduction": reproduction,
            "qualification_reproduction": qualification_ok,
            "single_run_firewall": single_run,
        },
    }
    output.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    metrics = result["metrics"]
    selection, sbc = metrics["selection"], metrics["adaptive_sbc"]
    summary.write_text("\n".join([
        "# V54 results: exact one-step expected information gain",
        "",
        f"Decision: `{result['decision']}`.",
        "",
        "V54 tests exact, one-step, open-loop assay selection for identifying `(program, theta)` while integrating out hidden dynamic state. It does not test reward planning, learned acquisition, language, or model access.",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Max candidate EIG error | {selection['maximum_absolute_candidate_eig_error']:.6g} |",
        f"| Optimal-set membership | {selection['optimal_set_membership_rate']:.3f} |",
        f"| Mean oracle EIG | {selection['mean_oracle_eig']:.6f} |",
        f"| Mean oracle advantage over random | {selection['mean_oracle_minus_uniform_random_eig']:.6f} |",
        f"| Informative record fraction | {selection['informative_record_fraction']:.3f} |",
        f"| Controls detected/dominated | {selection['controls']['detected_or_dominated']} |",
        f"| Adaptive SBC minimum p-value | {sbc['minimum_chi_square_p_value']:.6g} |",
        f"| Adaptive SBC max rank z | {sbc['maximum_absolute_rank_bin_z']:.6g} |",
        f"| Adaptive SBC max coverage z | {sbc['maximum_absolute_coverage_z']:.6g} |",
        "",
        f"All preregistered gates passed: `{str(result['qualification']['passed']).lower()}`.",
        f"Post-result audit: `{'pass' if audit['passed'] else 'fail'}`.",
    ]) + "\n")
    print(json.dumps(audit, indent=2, sort_keys=True))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
