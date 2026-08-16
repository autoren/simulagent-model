#!/usr/bin/env python3
"""Reproduce V35 assembly metrics, audit access, and summarize the result."""

from __future__ import annotations

import argparse
import json

import numpy as np

from audit_v32_factorized_semantics import read_rows
from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT
from v35_binding import qualification, score_assembly, select_binding_method, select_new_predicate_method, select_report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lock", default="configs/v35-binding-assembly-lock.json")
    parser.add_argument("--result", default="outputs/v35-binding-assembly/result.json")
    parser.add_argument("--audit", default="outputs/v35-binding-assembly/post-result-audit.json")
    parser.add_argument("--markdown", default="docs/v35-results.md")
    args = parser.parse_args()
    lock_path, result_path, audit_path, markdown_path = map(lambda value: (PROJECT_ROOT / value).resolve(), (args.lock, args.result, args.audit, args.markdown))
    lock, result = json.loads(lock_path.read_text()), json.loads(result_path.read_text())
    config = {**lock["config_payload"], "v32_config": lock["v32_config_payload"]}; errors = []
    if result["protocol_lock_sha256"] != file_sha256(lock_path):
        errors.append("V35 result does not bind protocol lock")
    for path, expected in lock["implementation"].items():
        if file_sha256(PROJECT_ROOT / path) != expected:
            errors.append(f"V35 implementation changed: {path}")
    metadata_path, prediction_path = PROJECT_ROOT / result["feature_metadata"], PROJECT_ROOT / result["predictions"]
    if file_sha256(metadata_path) != result["feature_metadata_sha256"]:
        errors.append("V35 feature metadata changed")
    if file_sha256(prediction_path) != result["predictions_sha256"]:
        errors.append("V35 predictions changed")
    rows = sorted(read_rows(PROJECT_ROOT / config["sourceCorpus"], tuple(config["allowedSplits"])), key=lambda row: row["id"])
    fit_indices = np.asarray([index for index, row in enumerate(rows) if row["split"] == "factor_fit"], dtype=np.int64)
    calibration_indices = np.asarray([index for index, row in enumerate(rows) if row["split"] == "factor_calibration"], dtype=np.int64)
    saved = np.load(prediction_path)
    if saved["record_ids"].tolist() != [row["id"] for row in rows]:
        errors.append("V35 prediction population differs")
    reproduced = {}
    for system in ("legacyAssembly", "modularAssembly"):
        predicate = saved[f"{system}_predicate"]; argument1 = saved[f"{system}_argument1"]; argument2 = saved[f"{system}_argument2"]
        bindings = [(int(first), None if int(second) < 0 else int(second)) for first, second in zip(argument1, argument2, strict=True)]
        reproduced[system] = {
            "fit": score_assembly(rows, fit_indices, predicate[fit_indices], [bindings[int(index)] for index in fit_indices], saved[f"{system}_lexical_sign"][fit_indices], saved[f"{system}_outer_operation"][fit_indices], config),
            "calibration": score_assembly(rows, calibration_indices, predicate[calibration_indices], [bindings[int(index)] for index in calibration_indices], saved[f"{system}_lexical_sign"][calibration_indices], saved[f"{system}_outer_operation"][calibration_indices], config),
        }
    selected_predicate = select_new_predicate_method(result["predicate_methods"])
    selected_binding = select_binding_method(result["binding_methods"])
    selected_qualification = qualification(reproduced["modularAssembly"]["calibration"], reproduced["legacyAssembly"]["calibration"], config)
    comparisons = {
        "metrics_reproduce": all(reproduced[system][split] == result["systems"][system][split] for system in reproduced for split in ("fit", "calibration")),
        "predicate_alpha_selections": all(select_report(value["cv_reports"]) == value["selected_cv"] for value in result["predicate_methods"].values()),
        "binding_alpha_selections": all(select_report(value["cv_reports"]) == value["selected_cv"] for value in result["binding_methods"].values()),
        "sign_alpha_selection": select_report(result["lexical_sign_method"]["cv_reports"]) == result["lexical_sign_method"]["selected_cv"],
        "selected_predicate": selected_predicate == result["fit_selected_predicate_method"],
        "selected_binding": selected_binding == result["fit_selected_binding_method"],
        "qualification": selected_qualification == result["qualification"],
    }
    if not all(comparisons.values()):
        errors.append("V35 result does not reproduce")
    access, limits = result["data_access"], config["limits"]
    access_checks = {
        "backbone_forward_passes": access["backbone_forward_passes"] == limits["backboneForwardPasses"],
        "ridge_training_fits": access["ridge_training_fits"] == limits["ridgeTrainingFits"],
        "no_calibration_selection": access["calibration_selections"] == 0,
        "no_v32_evaluation_records": access["v32_evaluation_records_read"] == 0,
        "no_v32_evaluation_features": access["v32_evaluation_features_read"] == 0,
        "no_v32_evaluation_predictions": access["v32_evaluation_predictions_read"] == 0,
        "no_adapter": access["adapter_training_runs"] == 0, "no_v28": access["v28_integration_replays"] == 0,
        "no_fresh_suite": access["fresh_suite_constructions"] == 0,
    }
    if not all(access_checks.values()):
        errors.append("V35 data access violates lock")
    audit = {"schema_version": 35, "experiment": "v35_post_result_audit", "result": str(result_path.relative_to(PROJECT_ROOT)), "result_sha256": file_sha256(result_path), "passed": not errors, "decision": "accept_v35_development_result" if not errors else "reject_v35_development_result", "errors": errors, "reproduction_checks": comparisons, "access_checks": access_checks, "fit_selected_predicate_method": selected_predicate, "fit_selected_binding_method": selected_binding, "qualification": selected_qualification}
    audit_path.parent.mkdir(parents=True, exist_ok=True); audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    lines = [
        "# V35 results: regularized atom binding and modular assembly", "", f"Decision: `{result['decision']}`.", "",
        "V35 is a fit/calibration-only development assembly. It does not reuse V32 evaluation, construct a fresh suite, or authorize V28.", "",
        "## Assembly", "", "| System | Predicate | Atom | Relation order | Sign | Operation | Compiled truth | Exact fact |", "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for system, label in (("legacyAssembly", "Legacy atom + repaired operation"), ("modularAssembly", "Fit-selected modular assembly")):
        metrics = reproduced[system]["calibration"]
        lines.append(f"| {label} | {metrics['predicate_accuracy']:.3f} | {metrics['atom_accuracy']:.3f} | {metrics['relation_order_accuracy']:.3f} | {metrics['lexical_sign_accuracy']:.3f} | {metrics['outer_operation_accuracy']:.3f} | {metrics['compiled_truth_accuracy']:.3f} | {metrics['compiled_exact_fact_accuracy']:.3f} |")
    lines.extend(["", "## Interpretation", "", f"Fit-only cross-validation selected predicate `{selected_predicate}` and binding `{selected_binding}`.", "", f"Modular exact-fact gain over the registered legacy assembly is {selected_qualification['calibration_exact_fact_gain_over_legacy']:+.3f}.", "", f"All qualification gates passed: `{str(selected_qualification['passed']).lower()}`.", "", f"Post-result integrity audit: `{'pass' if audit['passed'] else 'fail'}`."])
    markdown_path.write_text("\n".join(lines) + "\n")
    print(json.dumps(audit, indent=2, sort_keys=True))
    if not audit["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
