#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json

from evaluate_v49_partial import aggregate, qualification, read
from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", default="outputs/v49-passive-partial-observation/development/result.json")
    parser.add_argument("--output", default="outputs/v49-passive-partial-observation/post-result-audit.json")
    parser.add_argument("--summary", default="docs/v49-results.md")
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
        errors.append("V49 result is not bound to sealed corpus")
    if result.get("development_run_number") != 1 or attempt.get("status") != "completed":
        errors.append("V49 one-run state is invalid")
    mechanic_path = PROJECT_ROOT / result["mechanic_metrics"]
    if file_sha256(mechanic_path) != result["mechanic_metrics_sha256"]:
        errors.append("V49 mechanic metrics hash mismatch")
    details = read(mechanic_path)
    reproduced_all = aggregate(details)
    reproduced_evaluation = aggregate([row for row in details if row["split"] == "development_evaluation"])
    reproduced_q = qualification(reproduced_all, implementation["config_payload"]["gates"])
    reproduction = {
        "all_metrics": reproduced_all == result["metrics"]["all_mechanics"],
        "development_evaluation_metrics": reproduced_evaluation == result["metrics"]["development_evaluation"],
        "qualification": reproduced_q == result["qualification"],
    }
    if not all(reproduction.values()):
        errors.append("V49 reported result does not reproduce from sealed mechanic metrics")
    audit = {
        "schema_version": 49,
        "experiment": "v49_post_result_audit",
        "passed": not errors,
        "decision": "accept_v49_development" if not errors else "reject_v49_integrity",
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
        "# V49 results: passive partial observation",
        "",
        f"Decision: `{result['decision']}`.",
        "",
        "V49 is a non-final symbolic development test under known noiseless observation masks. It does not test language, active intervention selection, noisy sensors, continuous probabilities, or open ontologies.",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Mean conditional latent-suffix TV | {metrics['mean_conditional_latent_suffix_tv']:.6g} |",
        f"| Oracle-program partial TV | {metrics['oracle_program_partial_mean_tv']:.6g} |",
        f"| Held-out conditional log loss | {metrics['heldout_conditional_suffix_log_loss']:.6g} |",
        f"| MAP schema recovery | {metrics['map_schema_recovery']:.3f} |",
        f"| Mean target-program posterior | {metrics['mean_target_program_posterior']:.6f} |",
        f"| Probability MAE | {metrics['probability_parameter_mae']:.6g} |",
        f"| Partial-minus-full TV | {metrics['partial_minus_full_mean_tv']:.6g} |",
        f"| Partial-minus-full log loss | {metrics['partial_minus_full_log_loss']:.6g} |",
        f"| MAP-state collapse disadvantage | {metrics['map_latent_collapse_log_loss_disadvantage']:.6g} |",
        f"| History-ablation disadvantage | {metrics['observation_history_ablation_log_loss_disadvantage']:.6g} |",
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
