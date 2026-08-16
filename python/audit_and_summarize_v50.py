#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json

from evaluate_v50_history import aggregate, qualification, read
from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", default="outputs/v50-history-dependent-belief-filtering/development/result.json")
    parser.add_argument("--output", default="outputs/v50-history-dependent-belief-filtering/post-result-audit.json")
    parser.add_argument("--summary", default="docs/v50-results.md")
    args = parser.parse_args()
    result_path = (PROJECT_ROOT / args.result).resolve()
    output = (PROJECT_ROOT / args.output).resolve()
    summary = (PROJECT_ROOT / args.summary).resolve()
    result = json.loads(result_path.read_text())
    seal_path = PROJECT_ROOT / result["corpus_seal"]
    seal = json.loads(seal_path.read_text())
    implementation_path = PROJECT_ROOT / seal["implementation_lock"]
    implementation = json.loads(implementation_path.read_text())
    attempt_path = result_path.parent.parent / "development-attempt.json"
    attempt = json.loads(attempt_path.read_text()) if attempt_path.is_file() else {}
    errors = []
    if result["corpus_seal_sha256"] != file_sha256(seal_path):
        errors.append("V50 result is not bound to sealed corpus")
    if result.get("development_run_number") != 1 or attempt.get("status") != "completed":
        errors.append("V50 one-run state is invalid")
    mechanic_path = PROJECT_ROOT / result["mechanic_metrics"]
    if file_sha256(mechanic_path) != result["mechanic_metrics_sha256"]:
        errors.append("V50 mechanic metrics hash mismatch")
    details = read(mechanic_path)
    threshold = implementation["config_payload"]["historyDependenceContract"]["minimumOracleFullHistoryVsLatestOnlyTv"]
    reproduced_all = aggregate(details, threshold)
    reproduced_evaluation = aggregate(
        [row for row in details if row["split"] == "development_evaluation"], threshold
    )
    reproduced_q = qualification(reproduced_all, implementation["config_payload"]["gates"])
    reproduction = {
        "all_metrics": reproduced_all == result["metrics"]["all_mechanics"],
        "development_evaluation_metrics": reproduced_evaluation == result["metrics"]["development_evaluation"],
        "qualification": reproduced_q == result["qualification"],
    }
    if not all(reproduction.values()):
        errors.append("V50 reported result does not reproduce from sealed mechanic metrics")
    audit = {
        "schema_version": 50,
        "experiment": "v50_post_result_audit",
        "passed": not errors,
        "decision": "accept_v50_development" if not errors else "reject_v50_integrity",
        "errors": errors,
        "result": str(result_path.relative_to(PROJECT_ROOT)),
        "result_sha256": file_sha256(result_path),
        "scientific_decision": result["decision"],
        "integrity_checks": {
            "seal_bound": result["corpus_seal_sha256"] == file_sha256(seal_path),
            "implementation_bound": seal["implementation_lock_sha256"] == file_sha256(implementation_path),
            "one_run": result.get("development_run_number") == 1,
            "no_selection": result["data_access"]["selection_on_development_evaluation"] == 0,
            "no_model": result["data_access"]["model_forward_passes"] == 0,
            "no_training": result["data_access"]["adapter_training_runs"] == 0,
            "non_final": not result["authorization"]["final_evaluation"],
        },
        "reproduction_checks": reproduction,
    }
    output.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    metrics = result["metrics"]["all_mechanics"]
    lines = [
        "# V50 results: history-dependent belief filtering",
        "",
        f"Decision: `{result['decision']}`.",
        "",
        "V50 is a non-final exact symbolic development test. It tests temporal evidence retention under known value-independent masks and condition-matched scoring; it does not test language, active intervention selection, noisy sensors, continuous probabilities, or open ontologies.",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Mean complete-history suffix TV | {metrics['mean_complete_history_conditional_suffix_tv']:.6g} |",
        f"| Oracle-program suffix TV | {metrics['oracle_program_mean_tv']:.6g} |",
        f"| Complete-history condition-matched regret | {metrics['complete_history_condition_matched_regret']:.6g} |",
        f"| Oracle history-dependent query fraction | {metrics['oracle_history_dependent_query_fraction']:.3f} |",
        f"| Mean oracle full-history vs latest-only TV | {metrics['mean_oracle_full_history_vs_latest_only_tv']:.6g} |",
        f"| Latest-only log-loss disadvantage | {metrics['latest_only_log_loss_disadvantage']:.6g} |",
        f"| Time-shuffled log-loss disadvantage | {metrics['time_shuffled_log_loss_disadvantage']:.6g} |",
        f"| MAP-state collapse disadvantage | {metrics['map_latent_collapse_log_loss_disadvantage']:.6g} |",
        f"| Partial-minus-full condition-matched regret | {metrics['partial_minus_full_condition_matched_regret']:.6g} |",
        f"| Raw partial-minus-full log loss (non-gating) | {metrics['raw_partial_minus_full_log_loss_non_gating']:.6g} |",
        f"| Oracle conditional-entropy gap (non-gating) | {metrics['oracle_conditional_entropy_gap_non_gating']:.6g} |",
        f"| Calibration error | {metrics['calibration_error']:.6g} |",
        f"| MAP schema recovery | {metrics['map_schema_recovery']:.3f} |",
        f"| Mean target-program posterior | {metrics['mean_target_program_posterior']:.6f} |",
        f"| Probability MAE | {metrics['probability_parameter_mae']:.6g} |",
        "",
        f"All preregistered gates passed: `{str(result['qualification']['passed']).lower()}`.",
        f"Post-result integrity audit: `{'pass' if audit['passed'] else 'fail'}`.",
        "",
    ]
    summary.write_text("\n".join(lines))
    print(json.dumps(audit, indent=2, sort_keys=True))
    if not audit["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
