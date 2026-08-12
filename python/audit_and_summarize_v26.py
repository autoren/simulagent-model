"""Audit V26 result integrity and write its exposed-development report."""

from __future__ import annotations

import argparse
import json

from evaluate_v25_truth_hypotheses import gate_checks
from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT


def expected_decision(checks):
    if all(checks.values()):
        return "authorize_fresh_relational_surface_benchmark_design"
    if not checks["evaluation_truth"]:
        return "native_truth_decoder_insufficient_pivot_grounder_family_no_lora"
    return "repair_exact_graph_or_symbolic_composition_no_lora"


def candidate_map(path):
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return {
        (scene["scene_id"], row["evidence_id"]): row["candidate_id"]
        for scene in rows for row in scene["rows"]
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lock", default="configs/v26-native-truth-decoder-lock.json")
    parser.add_argument("--result", default="outputs/v26-native-truth-decoder/evaluation/result.json")
    parser.add_argument("--audit", default="outputs/v26-native-truth-decoder/post-result-audit.json")
    parser.add_argument("--markdown", default="docs/v26-results.md")
    args = parser.parse_args()
    lock_path = (PROJECT_ROOT / args.lock).resolve()
    result_path = (PROJECT_ROOT / args.result).resolve()
    lock = json.loads(lock_path.read_text())
    result = json.loads(result_path.read_text())
    attempt = json.loads((PROJECT_ROOT / "outputs/v26-native-truth-decoder/evaluation-attempt.json").read_text())
    reproduced = gate_checks(result["grounding"], result["integration"], lock["gates"]["development"])
    scores_path = PROJECT_ROOT / result["native_decoder_scores"]
    predictions_path = PROJECT_ROOT / result["grounding_predictions"]
    source_predictions = PROJECT_ROOT / lock["source"]["v24_predictions"]
    checks = {
        "lock_matches_result": result["protocol_lock_sha256"] == file_sha256(lock_path),
        "evaluation_attempt_completed_once": (
            attempt["attempt_number"] == 1 and attempt["status"] == "completed"
            and attempt["result_sha256"] == file_sha256(result_path)
        ),
        "score_artifact_intact": result["native_decoder_scores_sha256"] == file_sha256(scores_path),
        "prediction_artifact_intact": result["grounding_predictions_sha256"] == file_sha256(predictions_path),
        "v24_assignments_byte_equivalent": candidate_map(predictions_path) == candidate_map(source_predictions),
        "single_distinct_label_tokens": (
            set(result["label_token_ids"]) == {"A", "B", "C"}
            and len(set(result["label_token_ids"].values())) == 3
        ),
        "registered_fp32_score_used": result["observed_dtypes"]["fp32_direct_logits"] == "mlx.core.float32",
        "no_truncation": result["truncated_prompts"] == 0,
        "registered_forward_count": result["data_access"]["model_forward_passes"] == lock["limits"]["modelForwardPasses"],
        "all_conditions_present": set(result["integration"]) == set(lock["integration_conditions"]),
        "gates_reproduced": result["checks"] == reproduced,
        "pass_flag_reproduced": result["passed"] == all(reproduced.values()),
        "decision_reproduced": result["decision"] == expected_decision(reproduced),
        "zero_fit_and_selection": (
            result["data_access"]["head_fits"] == 0
            and result["data_access"]["threshold_fits"] == 0
            and result["data_access"]["hyperparameter_selections"] == 0
            and result["data_access"]["adapter_training_runs"] == 0
            and result["data_access"]["fresh_benchmark_records_read"] == 0
        ),
        "fresh_benchmark_not_constructed": not result["final_suite_constructed"],
        "lora_not_authorized": not result["lora_authorized"],
    }
    audit = {
        "schema_version": 26,
        "experiment": "v26_post_result_integrity_audit",
        "passed": all(checks.values()),
        "decision": "accept_v26_exposed_development_result" if all(checks.values()) else "quarantine_v26_result",
        "checks": checks,
        "reproduced_development_gates": reproduced,
    }
    audit_path = (PROJECT_ROOT / args.audit).resolve()
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")

    v25 = json.loads((PROJECT_ROOT / lock["source"]["v25_result"]).read_text())
    current = result["grounding"]["by_split"]["grounding_evaluation"]
    prior = v25["grounding"]["by_split"]["grounding_evaluation"]
    lines = [
        "# V26 results: full-depth native truth decoder",
        "",
        f"Decision: `{result['decision']}`.",
        "",
        "V26 is an exposed-data, zero-fit development evaluation. It kept every V24 candidate",
        "assignment fixed and selected truth status only by float32 A/B/C decoder logits from the",
        "frozen model's final layer.",
        "",
        "## Grounding",
        "",
        "| Split | Atom assignment | Relation order | Truth | Exact scene |",
        "|---|---:|---:|---:|---:|",
    ]
    for split, label in (
        ("grounding_fit", "Fit"), ("grounding_calibration", "Calibration"),
        ("grounding_evaluation", "Evaluation"),
    ):
        row = result["grounding"]["by_split"][split]
        lines.append(
            f"| {label} | {row['atom_assignment_accuracy']:.3f} | "
            f"{row['relation_argument_order_accuracy']:.3f} | {row['truth_status_accuracy']:.3f} | "
            f"{row['exact_scene_graph']:.3f} |"
        )
    lines.extend([
        "",
        f"V25 evaluation truth was {prior['truth_status_accuracy']:.3f}; V26 reaches {current['truth_status_accuracy']:.3f}.",
        "",
        "## Four-way integration",
        "",
        "| Support graph | Query graph | Transition-set exact | Target retention | Empty version space |",
        "|---|---|---:|---:|---:|",
    ])
    for name in lock["integration_conditions"]:
        row = result["integration"][name]
        lines.append(
            f"| {row['support_mode']} | {row['query_mode']} | {row['transition_set_exact_match']:.3f} | "
            f"{row['target_retention_rate']:.3f} | {row['empty_version_space_rate']:.3f} |"
        )
    lines.extend([
        "", "## Interpretation", "", result["interpretation"], "",
        "No head, threshold, model weight, matcher, proposal, ontology, DSL, or executor changed.",
        "Calibration selected nothing and no fresh benchmark was opened.", "",
        f"Post-result integrity audit: `{'pass' if audit['passed'] else 'fail'}`.", "",
    ])
    (PROJECT_ROOT / args.markdown).write_text("\n".join(lines))
    print(json.dumps(audit, indent=2, sort_keys=True))
    if not audit["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
