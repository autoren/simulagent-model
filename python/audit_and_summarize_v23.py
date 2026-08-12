"""Audit V23 integrity and write its compact research report."""

from __future__ import annotations

import argparse
import json

from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lock", default="configs/v23-probabilistic-support-lock.json")
    parser.add_argument("--result", default="outputs/v23-probabilistic-support/result.json")
    parser.add_argument("--audit", default="outputs/v23-probabilistic-support/post-result-audit.json")
    parser.add_argument("--markdown", default="docs/v23-results.md")
    args = parser.parse_args()
    lock_path = (PROJECT_ROOT / args.lock).resolve()
    result_path = (PROJECT_ROOT / args.result).resolve()
    lock = json.loads(lock_path.read_text())
    result = json.loads(result_path.read_text())
    attempt_path = PROJECT_ROOT / "outputs/v23-probabilistic-support-attempt.json"
    attempt = json.loads(attempt_path.read_text())
    reference_spec = lock["registered_reference"]
    reference = result["curves"]["grounding_evaluation"][str(reference_spec["branchBudget"])][str(reference_spec["credibleProgramMass"])]
    gates = lock["gates"]
    expected_checks = {
        "target_nonzero_retention": reference["target_nonzero_posterior_retention"] >= gates["referenceMinimumTargetNonzeroRetention"],
        "target_credible_retention": reference["target_credible_set_retention"] >= gates["referenceMinimumTargetCredibleRetention"],
        "empty_posterior": reference["empty_posterior_rate"] <= gates["referenceMaximumEmptyPosteriorRate"],
        "transition_set_exact": reference["transition_set_exact_match"] >= gates["referenceMinimumTransitionSetExact"],
        "excess_outcomes": reference["mean_excess_outcomes"] <= gates["referenceMaximumMeanExcessOutcomes"],
        "missing_target_outcomes": reference["missing_target_outcome_rate"] <= gates["referenceMaximumMissingTargetOutcomeRate"],
    }
    checks = {
        "lock_matches": result["protocol_lock_sha256"] == file_sha256(lock_path),
        "attempt_completed": attempt["status"] == "completed" and attempt["result_sha256"] == file_sha256(result_path),
        "all_registered_curve_cells_present": all(
            len(result["curves"][split]) == len(lock["branch_budgets"])
            and all(len(values) == len(lock["credible_program_masses"]) for values in result["curves"][split].values())
            for split in ("grounding_fit", "grounding_calibration", "grounding_evaluation")
        ),
        "registered_reference_matches_curve": result["registered_reference"] == reference,
        "gate_checks_reproduced": result["checks"] == expected_checks,
        "pass_flag_reproduced": result["passed"] == all(expected_checks.values()),
        "decision_reproduced": result["decision"] == "probabilistic_support_insufficient_revise_language_interface",
        "zero_model_refit_and_selection": (
            result["data_access"]["new_model_forward_passes"] == 0
            and result["data_access"]["new_linear_fits"] == 0
            and result["data_access"]["hyperparameter_selections"] == 0
            and result["data_access"]["adapter_training_runs"] == 0
        ),
    }
    audit = {
        "schema_version": 23,
        "experiment": "v23_post_result_integrity_audit",
        "passed": all(checks.values()),
        "decision": "accept_v23_negative_result" if all(checks.values()) else "quarantine_v23_result",
        "checks": checks,
        "registered_gate_checks": expected_checks,
    }
    audit_path = (PROJECT_ROOT / args.audit).resolve()
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")

    curve = result["curves"]["grounding_evaluation"]
    lines = [
        "# V23 results: probabilistic relational support replay",
        "",
        f"Decision: `{result['decision']}`.",
        "",
        "V23 is an exposed-data development diagnostic, not a holdout or final result. It reused the",
        "frozen V22r2 features and heads with zero model forwards, fits, or hyperparameter selections.",
        "",
        "## Registered reference",
        "",
        f"At 64 graph branches and 95% credible program mass, target nonzero retention was {reference['target_nonzero_posterior_retention']:.3f},",
        f"credible-set retention was {reference['target_credible_set_retention']:.3f}, and empty posteriors were {reference['empty_posterior_rate']:.3f}.",
        f"However, transition-set exact match was only {reference['transition_set_exact_match']:.3f}, with {reference['mean_excess_outcomes']:.3f}",
        f"excess outcomes per query and a median of {reference['median_credible_programs']:.1f} credible programs.",
        "",
        "## Evaluation curve",
        "",
        "| Graph branches | Program mass | Exact | Target nonzero | Target credible | Empty | Excess | Missing |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for budget in lock["branch_budgets"]:
        for mass in lock["credible_program_masses"]:
            row = curve[str(budget)][str(mass)]
            lines.append(
                f"| {budget} | {mass:.2f} | {row['transition_set_exact_match']:.3f} | "
                f"{row['target_nonzero_posterior_retention']:.3f} | {row['target_credible_set_retention']:.3f} | "
                f"{row['empty_posterior_rate']:.3f} | {row['mean_excess_outcomes']:.3f} | "
                f"{row['missing_target_outcome_rate']:.3f} |"
            )
    lines.extend([
        "", "## Interpretation", "",
        "Uncertainty propagation repairs catastrophic pruning but not identification. As graph coverage",
        "rises, many incorrect programs acquire nonzero likelihood; credible unions then return nearly",
        "the full outcome vocabulary. No registered cell combines high target retention with precise",
        "answers. This is an anti-widening failure, not a probabilistic repair.",
        "",
        "The next development direction is a candidate-conditioned frozen cross-encoder: compare each",
        "evidence clause directly with a small set of atom and truth hypotheses, with explicit ordered",
        "relation arguments. V22r2's top-3 matcher can be used only as a recall-preserving proposal stage.",
        "A fresh surface benchmark is required after that interface is fixed because all V22r2 splits",
        "are exposed. LoRA, a final suite, grammar expansion, and a neural challenger remain unauthorized.",
        "",
        f"Post-result integrity audit: `{'pass' if audit['passed'] else 'fail'}`.", "",
    ])
    markdown_path = (PROJECT_ROOT / args.markdown).resolve()
    markdown_path.write_text("\n".join(lines))
    print(json.dumps(audit, indent=2, sort_keys=True))
    if not audit["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
