"""Audit V24 result integrity and write its compact exposed-development report."""

from __future__ import annotations

import argparse
import json

from evaluate_v24_cross_encoder import gate_checks
from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT


def expected_decision(checks: dict[str, bool]) -> str:
    if all(checks.values()):
        return "authorize_fresh_relational_surface_benchmark_design"
    if not checks["evaluation_atom_assignment"] or not checks["evaluation_relation_order"]:
        return "candidate_conditioned_comparison_insufficient_no_lora"
    if not checks["evaluation_truth"]:
        return "factor_truth_semantics_before_fresh_benchmark_no_lora"
    return "repair_symbolic_composition_before_fresh_benchmark_no_lora"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lock", default="configs/v24-cross-encoder-lock.json")
    parser.add_argument("--metadata", default="outputs/v24-cross-encoder/features/metadata.json")
    parser.add_argument("--result", default="outputs/v24-cross-encoder/evaluation/result.json")
    parser.add_argument("--audit", default="outputs/v24-cross-encoder/post-result-audit.json")
    parser.add_argument("--markdown", default="docs/v24-results.md")
    args = parser.parse_args()
    lock_path = (PROJECT_ROOT / args.lock).resolve()
    metadata_path = (PROJECT_ROOT / args.metadata).resolve()
    result_path = (PROJECT_ROOT / args.result).resolve()
    lock = json.loads(lock_path.read_text())
    metadata = json.loads(metadata_path.read_text())
    result = json.loads(result_path.read_text())
    feature_attempt = json.loads((PROJECT_ROOT / "outputs/v24-cross-encoder/feature-extraction-attempt.json").read_text())
    evaluation_attempt = json.loads((PROJECT_ROOT / "outputs/v24-cross-encoder/evaluation-attempt.json").read_text())
    reproduced_checks = gate_checks(
        result["grounding"], result["integration"], lock["gates"]["development"]
    )
    feature_path = PROJECT_ROOT / metadata["feature_artifact"]
    heads_path = PROJECT_ROOT / result["heads_artifact"]
    predictions_path = PROJECT_ROOT / result["grounding_predictions"]
    expected_conditions = set(lock["integration_conditions"])
    checks = {
        "lock_matches_features_and_result": (
            metadata["protocol_lock_sha256"] == file_sha256(lock_path)
            and result["protocol_lock_sha256"] == file_sha256(lock_path)
        ),
        "feature_attempt_completed_once": (
            feature_attempt["attempt_number"] == 1
            and feature_attempt["status"] == "completed"
            and feature_attempt["metadata_sha256"] == file_sha256(metadata_path)
        ),
        "evaluation_attempt_completed_once": (
            evaluation_attempt["attempt_number"] == 1
            and evaluation_attempt["status"] == "completed"
            and evaluation_attempt["result_sha256"] == file_sha256(result_path)
        ),
        "feature_artifact_intact": metadata["feature_artifact_sha256"] == file_sha256(feature_path),
        "heads_artifact_intact": result["heads_artifact_sha256"] == file_sha256(heads_path),
        "prediction_artifact_intact": result["grounding_predictions_sha256"] == file_sha256(predictions_path),
        "registered_forward_budget_respected": (
            metadata["new_model_forward_passes"] == lock["pre_extraction_audit"]["budget"]["planned_model_forwards"]
            and metadata["new_model_forward_passes"] <= lock["gates"]["preExtraction"]["maximumNewModelForwardPasses"]
        ),
        "no_truncation": metadata["truncated_prompts"] == 0,
        "all_conditions_present": set(result["integration"]) == expected_conditions,
        "gate_checks_reproduced": result["checks"] == reproduced_checks,
        "pass_flag_reproduced": result["passed"] == all(reproduced_checks.values()),
        "decision_reproduced": result["decision"] == expected_decision(reproduced_checks),
        "one_shot_limits_respected": (
            result["data_access"]["match_head_fits"] == 1
            and result["data_access"]["truth_head_fits"] == 1
            and result["data_access"]["integration_evaluations"] == 1
            and result["data_access"]["hyperparameter_selections"] == 0
            and result["data_access"]["adapter_training_runs"] == 0
            and result["data_access"]["fresh_benchmark_records_read"] == 0
        ),
        "proposal_coverage_matches_audit": (
            result["evaluation_proposal_coverage"]
            == lock["pre_extraction_audit"]["proposal"]["gold_coverage_by_split_and_role"]["grounding_evaluation"]
        ),
        "fresh_benchmark_not_constructed": not result["final_suite_constructed"],
        "lora_not_authorized": not result["lora_authorized"],
    }
    audit = {
        "schema_version": 24,
        "experiment": "v24_post_result_integrity_audit",
        "passed": all(checks.values()),
        "decision": "accept_v24_exposed_development_result" if all(checks.values()) else "quarantine_v24_result",
        "checks": checks,
        "reproduced_development_gates": reproduced_checks,
    }
    audit_path = (PROJECT_ROOT / args.audit).resolve()
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")

    grounding = result["grounding"]["by_split"]
    evaluation = grounding["grounding_evaluation"]
    calibration = grounding["grounding_calibration"]
    integration = result["integration"]
    lines = [
        "# V24 results: candidate-conditioned frozen relational grounding",
        "",
        f"Decision: `{result['decision']}`.",
        "",
        "V24 is an exposed-data development experiment, not a holdout or final result. The model,",
        "layer, ontology, candidate atoms, DSL, executor, proposal count, heads, and gates were frozen",
        "before the single feature extraction and head fit.",
        "",
        "## Proposal and extraction",
        "",
        f"The top-three-plus-hard proposal graph retained {result['evaluation_proposal_coverage']['support']:.3f} of evaluation-support",
        f"gold edges and {result['evaluation_proposal_coverage']['query']:.3f} of evaluation-query gold edges.",
        f"The frozen 4B model executed {metadata['new_model_forward_passes']} candidate-conditioned forwards with zero truncation.",
        "",
        "## Grounding",
        "",
        "| Split | Atom assignment | Relation order | Truth | Exact scene |",
        "|---|---:|---:|---:|---:|",
    ]
    for split, label in (
        ("grounding_fit", "Fit"),
        ("grounding_calibration", "Calibration"),
        ("grounding_evaluation", "Evaluation"),
    ):
        row = grounding[split]
        lines.append(
            f"| {label} | {row['atom_assignment_accuracy']:.3f} | "
            f"{row['relation_argument_order_accuracy']:.3f} | {row['truth_status_accuracy']:.3f} | "
            f"{row['exact_scene_graph']:.3f} |"
        )
    lines.extend([
        "",
        "## Four-way integration on evaluation episodes",
        "",
        "| Support graph | Query graph | Transition-set exact | Target retention | Empty version space |",
        "|---|---|---:|---:|---:|",
    ])
    for name in lock["integration_conditions"]:
        row = integration[name]
        lines.append(
            f"| {row['support_mode']} | {row['query_mode']} | {row['transition_set_exact_match']:.3f} | "
            f"{row['target_retention_rate']:.3f} | {row['empty_version_space_rate']:.3f} |"
        )
    lines.extend([
        "",
        "## Interpretation",
        "",
        result["interpretation"],
        "",
        "Calibration was report-only and selected no model, feature, threshold, regularization, or",
        "proposal policy. Regardless of the result, V24 does not itself support a final scientific claim.",
        "",
        f"Post-result integrity audit: `{'pass' if audit['passed'] else 'fail'}`.",
        "",
    ])
    markdown_path = (PROJECT_ROOT / args.markdown).resolve()
    markdown_path.write_text("\n".join(lines))
    print(json.dumps(audit, indent=2, sort_keys=True))
    if not audit["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
