#!/usr/bin/env python3
"""Reproduce the frozen V37 validation metrics and write the durable result report."""

from __future__ import annotations

import argparse
import json

from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT
from v37_semantic import qualification, score_semantics, select_method


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", default="outputs/v37-semantic-invariance/evaluation/result.json")
    parser.add_argument("--audit", default="outputs/v37-semantic-invariance/post-result-audit.json")
    parser.add_argument("--markdown", default="docs/v37-results.md")
    args = parser.parse_args()
    result_path, audit_path, markdown_path = (
        (PROJECT_ROOT / value).resolve() for value in (args.result, args.audit, args.markdown)
    )
    result = json.loads(result_path.read_text())
    feature_lock_path = PROJECT_ROOT / result["features_lock"]
    feature_lock = json.loads(feature_lock_path.read_text())
    seal_path = PROJECT_ROOT / feature_lock["corpus_seal"]
    seal = json.loads(seal_path.read_text())
    implementation_path = PROJECT_ROOT / seal["implementation_lock"]
    implementation = json.loads(implementation_path.read_text())
    config, v32_config = implementation["config_payload"], implementation["v32_config_payload"]
    rows = sorted(read_jsonl(PROJECT_ROOT / seal["corpora"]["validation"]["path"]), key=lambda row: row["id"])
    selected_rows = read_jsonl(PROJECT_ROOT / result["selected_predictions"])
    baseline_rows = read_jsonl(PROJECT_ROOT / result["baseline_predictions"])
    errors: list[str] = []
    if file_sha256(PROJECT_ROOT / result["selected_predictions"]) != result["selected_predictions_sha256"]:
        errors.append("V37 selected predictions changed")
    if file_sha256(PROJECT_ROOT / result["baseline_predictions"]) != result["baseline_predictions_sha256"]:
        errors.append("V37 baseline predictions changed")
    selected_lookup = {row["id"]: row["prediction"] for row in selected_rows}
    baseline_lookup = {row["id"]: row["prediction"] for row in baseline_rows}
    selected_metrics = score_semantics(rows, [selected_lookup[row["id"]] for row in rows], v32_config)
    baseline_metrics = score_semantics(rows, [baseline_lookup[row["id"]] for row in rows], v32_config)
    reproduced_qualification = qualification(selected_metrics, baseline_metrics, config)
    reproduction = {
        "selected_metrics": selected_metrics == result["selected_validation"],
        "baseline_metrics": baseline_metrics == result["frozen_v36_baseline"],
        "qualification": reproduced_qualification == result["qualification"],
        "decision": reproduced_qualification["decision"] == result["decision"],
    }
    if not all(reproduction.values()):
        errors.append("V37 saved predictions do not reproduce the result")
    selection_checks = {}
    for component, reports in result["cv_reports"].items():
        expected = select_method(reports, config["interfaces"]["methods"])
        selection_checks[component] = expected == result["component_selection"][component]
    if not all(selection_checks.values()):
        errors.append("V37 selected method does not follow fit CV")
    access = result["data_access"]
    access_checks = {
        "fit_population": access["fit_records_used"] == 400,
        "validation_population": access["validation_records_scored"] == 360,
        "one_validation": access["validation_evaluations"] == 1,
        "no_validation_selection": access["selection_on_validation"] == 0,
        "no_threshold_change": access["threshold_changes"] == 0,
        "no_evaluation_forwards": access["model_forward_passes"] == 0,
        "no_v32_calibration": access["v32_calibration_records_read"] == 0,
        "no_v32_evaluation": access["v32_evaluation_records_read"] == 0,
        "no_v28": access["v28_runs"] == 0,
        "no_adapter": access["adapter_training_runs"] == 0,
    }
    if not all(access_checks.values()):
        errors.append("V37 evaluation access ledger violates the lock")
    feature_metadata = json.loads((PROJECT_ROOT / feature_lock["feature_metadata"]).read_text())
    chain_checks = {
        "implementation_bound": seal["implementation_lock_sha256"] == file_sha256(implementation_path),
        "corpora_sealed": all(
            file_sha256(PROJECT_ROOT / metadata["path"]) == metadata["sha256"]
            for metadata in seal["corpora"].values()
        ),
        "exact_forward_budget": feature_metadata["backbone_forward_passes"] == 6840,
        "zero_truncation": feature_metadata["truncated_prompts"] == 0,
        "features_bound": result["features_lock_sha256"] == file_sha256(feature_lock_path),
    }
    if not all(chain_checks.values()):
        errors.append("V37 lock chain failed")
    audit = {
        "schema_version": 37,
        "experiment": "v37_post_result_audit",
        "passed": not errors,
        "decision": "accept_v37_result" if not errors else "reject_v37_result",
        "errors": errors,
        "result": str(result_path.relative_to(PROJECT_ROOT)),
        "result_sha256": file_sha256(result_path),
        "reproduction_checks": reproduction,
        "selection_checks": selection_checks,
        "access_checks": access_checks,
        "chain_checks": chain_checks,
        "scientific_decision": result["decision"],
    }
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    selected = result["selected_validation"]
    baseline = result["frozen_v36_baseline"]
    qual = result["qualification"]
    lines = [
        "# V37 results: candidate-conditioned semantic invariance", "",
        f"Decision: `{result['decision']}`.", "",
        "V37 is a development-only semantic-interface result. It does not use V32 calibration/evaluation, V28, adapter training, or an end-to-end relational suite.", "",
        "## Fresh validation", "",
        "| Metric | Selected interface | Frozen V36 interface |", "|---|---:|---:|",
        f"| Lexical sign | {selected['lexical_sign_accuracy']:.3f} | {baseline['lexical_sign_accuracy']:.3f} |",
        f"| Outer operation | {selected['outer_operation_accuracy']:.3f} | {baseline['outer_operation_accuracy']:.3f} |",
        f"| Compiled truth | {selected['compiled_truth_accuracy']:.3f} | {baseline['compiled_truth_accuracy']:.3f} |",
        f"| Worst operation | {selected['worst_operation_accuracy']:.3f} | {baseline['worst_operation_accuracy']:.3f} |",
        f"| Worst surface family truth | {selected['worst_surface_family_truth_accuracy']:.3f} | {baseline['worst_surface_family_truth_accuracy']:.3f} |",
        f"| Distractor truth | {selected['distractor_truth_accuracy']:.3f} | {baseline['distractor_truth_accuracy']:.3f} |",
        f"| Negative-composition truth | {selected['negative_composition_truth_accuracy']:.3f} | {baseline['negative_composition_truth_accuracy']:.3f} |",
        "", "## Selection and interpretation", "",
        f"Fit-only sign selection: `{result['component_selection']['lexical_sign']['method']}` with alpha `{result['component_selection']['lexical_sign']['alpha']}`.", "",
        f"Fit-only operation selection: `{result['component_selection']['outer_operation']['method']}` with alpha `{result['component_selection']['outer_operation']['alpha']}`.", "",
        f"Compiled-truth gain over the untouched V36 interface: {qual['compiled_truth_gain_over_frozen_v36']:+.3f}. All registered gates passed: `{str(qual['passed']).lower()}`.", "",
        f"Post-result integrity audit: `{'pass' if audit['passed'] else 'fail'}`.",
    ]
    markdown_path.write_text("\n".join(lines) + "\n")
    print(json.dumps(audit, indent=2, sort_keys=True))
    if not audit["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
