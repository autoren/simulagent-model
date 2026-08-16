#!/usr/bin/env python3
"""Reproduce, integrity-audit, and summarize the bounded V33 development result."""

from __future__ import annotations

import argparse
import json

from audit_v32_factorized_semantics import read_rows
from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT
from v33_development import (
    score_development, select_qualified_system, select_search_checkpoint, system_qualification,
)


def jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lock", default="configs/v33-development-adequacy-lock.json")
    parser.add_argument("--result", default="outputs/v33-development-adequacy/result.json")
    parser.add_argument("--audit", default="outputs/v33-development-adequacy/post-result-audit.json")
    parser.add_argument("--markdown", default="docs/v33-results.md")
    args = parser.parse_args()
    lock_path, result_path = (PROJECT_ROOT / args.lock).resolve(), (PROJECT_ROOT / args.result).resolve()
    lock, result = json.loads(lock_path.read_text()), json.loads(result_path.read_text())
    config, v32_config, errors = lock["config_payload"], lock["v32_config_payload"], []
    if result["protocol_lock_sha256"] != file_sha256(lock_path): errors.append("V33 protocol-lock hash mismatch")
    for path, expected in lock["implementation"].items():
        if file_sha256(PROJECT_ROOT / path) != expected: errors.append(f"V33 implementation changed: {path}")
    root = result_path.parent
    for name, expected in result["parameter_artifacts"].items():
        if file_sha256(root / name) != expected: errors.append(f"V33 parameter/ledger artifact changed: {name}")
    for name, expected in result["prediction_artifacts"].items():
        if file_sha256(root / name) != expected: errors.append(f"V33 prediction artifact changed: {name}")
    search_path = PROJECT_ROOT / result["search"]
    if file_sha256(search_path) != result["search_sha256"]: errors.append("V33 search artifact changed")
    search = json.loads(search_path.read_text())
    selected = {
        objective: select_search_checkpoint(objective, [row for row in search["reports"] if row["objective"] == objective], config)
        for objective in config["search"]["objectives"]
    }
    if selected != result["selected_search_configurations"]: errors.append("V33 search selection does not reproduce")
    rows = sorted(read_rows(PROJECT_ROOT / config["sourceCorpus"], tuple(config["allowedSplits"])), key=lambda row: row["id"])
    split_rows = {split: [row for row in rows if row["split"] == split] for split in config["allowedSplits"]}
    split_label = {"fit": "factor_fit", "calibration": "factor_calibration"}
    system_values = {name: {} for name in config["confirmation"]["candidateSystems"]}
    for seed in config["confirmation"]["seeds"]:
        for system in system_values:
            system_values[system][str(seed)] = {}
            for split, source_split in split_label.items():
                path = root / f"confirmation/seed-{seed}/{system}-{split}-predictions.jsonl"
                system_values[system][str(seed)][split] = score_development(
                    split_rows[source_split], jsonl(path), v32_config
                )
    qualification = {
        name: system_qualification(system_values[name], config)
        for name in config["qualification"]["eligibleSystems"]
    }
    selected_system, selection = select_qualified_system(qualification, config)
    comparisons = {
        "selected_search_configurations": selected == result["selected_search_configurations"],
        "system_seed_metrics": all(
            values == result["systems"][system]["seeds"] for system, values in system_values.items()
        ),
        "qualification": qualification == result["qualification"],
        "selection": selection == result["selection"],
        "selected_system": selected_system == result["selected_system"],
        "development_qualified": (selected_system is not None) == result["development_qualified"],
    }
    if not all(comparisons.values()): errors.append("V33 result does not reproduce from development predictions")
    access = result["data_access"]
    limits = config["limits"]
    access_checks = {
        "search_training_paths": access["search_training_paths"] == limits["searchTrainingPaths"],
        "search_checkpoint_evaluations": access["search_checkpoint_evaluations"] == limits["searchCheckpointEvaluations"],
        "confirmation_training_runs": access["confirmation_training_runs"] == limits["confirmationTrainingRuns"],
        "no_v32_evaluation_records": access["v32_evaluation_records_read"] == 0,
        "no_v32_evaluation_features": access["v32_evaluation_features_read"] == 0,
        "no_v32_evaluation_predictions": access["v32_evaluation_predictions_read"] == 0,
        "no_backbone": access["backbone_forward_passes"] == 0,
        "no_v28": access["v28_integration_replays"] == 0,
        "no_fresh_suite": access["fresh_suite_constructions"] == 0,
    }
    if not all(access_checks.values()): errors.append("V33 data-access ledger violates the lock")
    audit = {
        "schema_version": 33, "experiment": "v33_post_result_audit",
        "result": str(result_path.relative_to(PROJECT_ROOT)), "result_sha256": file_sha256(result_path),
        "passed": not errors, "decision": "accept_v33_development_result" if not errors else "reject_v33_development_result",
        "errors": errors, "reproduction_checks": comparisons, "access_checks": access_checks,
        "selected_system": selected_system, "development_qualified": selected_system is not None,
        "fresh_suite_preregistration_authorized": selected_system is not None,
        "v32_evaluation_reuse_authorized": False, "v28_authorized": False,
    }
    audit_path = (PROJECT_ROOT / args.audit).resolve()
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    independent, joint = qualification["independentCompiled"], qualification["jointCompiled"]
    selected_rows = [
        f"| {name} | {value['learning_rate']:.4g} | {value['epoch']} | {value['fit'][config['search']['selectionPrimaryMetric'][name]]:.3f} | {value['calibration'][config['search']['selectionPrimaryMetric'][name]]:.3f} |"
        for name, value in selected.items()
    ]
    lines = [
        "# V33 development adequacy", "", "## Verdict", "",
        f"Development-qualified interface: `{selected_system or 'none'}`. Diagnosis: `{result['diagnosis']}`.",
        "This is a fit/calibration development result, not a sealed generalization claim.", "",
        "## Selected learning-curve configurations", "",
        "| Objective | Learning rate | Epoch | Fit primary | Calibration primary |", "|---|---:|---:|---:|---:|", *selected_rows, "",
        "## Three-seed confirmation", "",
        "| Candidate | Fit atom | Fit sign | Fit operation | Fit compiled fact | Calibration atom | Calibration sign | Calibration operation | Calibration compiled fact | Pass |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|:---:|",
        f"| Independent compiled | {independent['fit_mean']['atom_exact_accuracy']:.3f} | {independent['fit_mean']['lexical_sign_accuracy']:.3f} | {independent['fit_mean']['outer_operation_accuracy']:.3f} | {independent['fit_mean']['compiled_exact_fact_accuracy']:.3f} | {independent['calibration_mean']['atom_exact_accuracy']:.3f} | {independent['calibration_mean']['lexical_sign_accuracy']:.3f} | {independent['calibration_mean']['outer_operation_accuracy']:.3f} | {independent['calibration_mean']['compiled_exact_fact_accuracy']:.3f} | {'yes' if independent['passed'] else 'no'} |",
        f"| Joint compiled | {joint['fit_mean']['atom_exact_accuracy']:.3f} | {joint['fit_mean']['lexical_sign_accuracy']:.3f} | {joint['fit_mean']['outer_operation_accuracy']:.3f} | {joint['fit_mean']['compiled_exact_fact_accuracy']:.3f} | {joint['calibration_mean']['atom_exact_accuracy']:.3f} | {joint['calibration_mean']['lexical_sign_accuracy']:.3f} | {joint['calibration_mean']['outer_operation_accuracy']:.3f} | {joint['calibration_mean']['compiled_exact_fact_accuracy']:.3f} | {'yes' if joint['passed'] else 'no'} |",
        "", "## Firewall", "", "V32 evaluation reuse: `none`.", "Backbone passes: `0`.", "V28 replays: `0`.", "Fresh-suite constructions: `0`.", f"Post-result audit: `{'pass' if audit['passed'] else 'fail'}`.", "",
    ]
    (PROJECT_ROOT / args.markdown).write_text("\n".join(lines))
    print(json.dumps(audit, indent=2, sort_keys=True))
    if not audit["passed"]: raise SystemExit(1)


if __name__ == "__main__": main()
