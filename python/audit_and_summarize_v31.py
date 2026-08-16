#!/usr/bin/env python3
"""Reproduce, integrity-audit, and summarize the one-shot V31 result."""

from __future__ import annotations

import argparse
import json

from audit_v31_signed_fact_adaptation import read_rows
from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT
from v31_evaluation import family_bootstrap_delta, summarize_seed, system_summary


def jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trained-lock", default="configs/v31-trained-systems-lock.json")
    parser.add_argument("--result", default="outputs/v31-signed-fact-adaptation/sealed-evaluation/result.json")
    parser.add_argument("--audit", default="outputs/v31-signed-fact-adaptation/post-result-audit.json")
    parser.add_argument("--markdown", default="docs/v31-results.md")
    args = parser.parse_args()
    trained_path = (PROJECT_ROOT / args.trained_lock).resolve()
    result_path = (PROJECT_ROOT / args.result).resolve()
    trained = json.loads(trained_path.read_text())
    protocol_path = PROJECT_ROOT / trained["protocol_lock"]
    protocol = json.loads(protocol_path.read_text())
    config = protocol["config_payload"]
    result = json.loads(result_path.read_text())
    errors = []
    if result["protocol_lock_sha256"] != file_sha256(protocol_path):
        errors.append("V31 result protocol-lock hash mismatch")
    if result["trained_system_lock_sha256"] != file_sha256(trained_path):
        errors.append("V31 result trained-system-lock hash mismatch")
    for path, expected in protocol["implementation"].items():
        if file_sha256(PROJECT_ROOT / path) != expected:
            errors.append(f"V31 implementation changed: {path}")
    evaluation_root = result_path.parent
    for name, expected in result["prediction_artifacts"].items():
        if file_sha256(evaluation_root / name) != expected:
            errors.append(f"V31 prediction artifact changed: {name}")
    if file_sha256(PROJECT_ROOT / result["evaluation_features"]) != result["evaluation_features_sha256"]:
        errors.append("V31 sealed feature artifact changed")
    rows = sorted(read_rows(
        PROJECT_ROOT / protocol["source"]["corpus"], ("adaptation_evaluation",)
    ), key=lambda row: row["id"])
    zero_predictions = jsonl(evaluation_root / "zero-shot-reference-predictions.jsonl")
    zero = summarize_seed(rows, zero_predictions, config, apply_gates=False)
    frozen_predictions, lora_predictions = {}, {}
    frozen_seeds, lora_seeds = {}, {}
    for seed in config["training"]["seeds"]:
        frozen = jsonl(evaluation_root / f"frozen-seed-{seed}-predictions.jsonl")
        lora = jsonl(evaluation_root / f"lora-seed-{seed}-predictions.jsonl")
        frozen_predictions[str(seed)] = frozen
        lora_predictions[str(seed)] = lora
        frozen_seeds[str(seed)] = summarize_seed(rows, frozen, config, apply_gates=True)
        lora_seeds[str(seed)] = summarize_seed(rows, lora, config, apply_gates=True)
    frozen_summary = system_summary(frozen_seeds, config)
    lora_summary = system_summary(lora_seeds, config)
    paired = family_bootstrap_delta(rows, frozen_predictions, lora_predictions, config)
    fact_delta = lora_summary["mean"]["exact_signed_fact_accuracy"] - frozen_summary["mean"]["exact_signed_fact_accuracy"]
    scene_delta = lora_summary["mean"]["exact_scene_accuracy"] - frozen_summary["mean"]["exact_scene_accuracy"]
    advantage_gates = config["gates"]["loraMaterialAdvantage"]
    advantage_checks = {
        "exact_signed_fact_delta": fact_delta >= advantage_gates["minimumMeanExactSignedFactDelta"],
        "exact_scene_delta": scene_delta >= advantage_gates["minimumMeanExactSceneDelta"],
        "paired_family_bootstrap_lower_bound": paired["bootstrap_95_interval"][0] > advantage_gates["minimumPairedFamilyBootstrapLowerBound"],
    }
    material = all(advantage_checks.values())
    if frozen_summary["passed"]:
        selected = "lora_readout" if lora_summary["passed"] and material else "frozen_readout"
    elif lora_summary["passed"]:
        selected = "lora_readout"
    else:
        selected = None
    decision = {
        "frozen_readout": "frozen_pass_selected_no_material_lora_advantage",
        "lora_readout": "lora_pass_selected_representation_adaptation_supported",
        None: "both_learned_systems_fail_stop_no_v28_replay",
    }[selected]
    comparisons = {
        "zero_shot_reference": zero == result["zero_shot_reference"],
        "frozen_readout": frozen_summary == result["frozen_readout"],
        "lora_readout": lora_summary == result["lora_readout"],
        "paired_family": paired == result["lora_minus_frozen"]["paired_family"],
        "fact_delta": fact_delta == result["lora_minus_frozen"]["mean_exact_signed_fact_delta"],
        "scene_delta": scene_delta == result["lora_minus_frozen"]["mean_exact_scene_delta"],
        "advantage_checks": advantage_checks == result["lora_minus_frozen"]["checks"],
        "material_advantage": material == result["lora_minus_frozen"]["material_advantage"],
        "selected_system": selected == result["selected_system"],
        "decision": decision == result["decision"],
        "integration_authorization": (selected is not None) == result["v28_integration_authorized"],
    }
    if not all(comparisons.values()):
        errors.append("V31 result does not exactly reproduce from sealed predictions")
    access = result["data_access"]
    expected_passes = (
        protocol["planned_evaluation"]["zero_shot_forward_passes"]
        + protocol["planned_evaluation"]["frozen_feature_forward_passes"]
        + protocol["planned_evaluation"]["lora_forward_passes"]
    )
    observed_passes = (
        access["zero_shot_model_forward_passes"] + access["frozen_feature_model_forward_passes"]
        + access["lora_model_forward_passes"]
    )
    if observed_passes != expected_passes:
        errors.append("V31 sealed forward-pass ledger differs from lock")
    if access["sealed_evaluations"] != 1 or access["seed_selections"] or access["checkpoint_selections"] or access["hyperparameter_selections"]:
        errors.append("V31 sealed access/selection ledger violates protocol")
    audit = {
        "schema_version": 31, "experiment": "v31_post_result_integrity_audit",
        "result": str(result_path.relative_to(PROJECT_ROOT)),
        "result_sha256": file_sha256(result_path),
        "passed": not errors,
        "decision": "accept_v31_language_result" if not errors else "reject_v31_language_result",
        "errors": errors, "reproduction_checks": comparisons,
        "selected_system": selected, "language_passed": selected is not None,
        "v28_integration_authorized": selected is not None,
        "observed_model_forward_passes": observed_passes,
        "expected_model_forward_passes": expected_passes,
    }
    audit_path = (PROJECT_ROOT / args.audit).resolve()
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    lines = [
        "# V31 matched-head signed-fact adaptation", "",
        "## Verdict", "",
        (
            f"The frozen readout {'passed' if frozen_summary['passed'] else 'failed'} and the LoRA readout "
            f"{'passed' if lora_summary['passed'] else 'failed'} the preregistered language gates. "
            f"Selected system: `{selected or 'none'}`."
        ), "",
        "## Sealed evaluation", "",
        "| System | Predicate | Arg 1 | Relation order | Truth | Exact fact | Exact scene | Pass |",
        "|---|---:|---:|---:|---:|---:|---:|:---:|",
        f"| Zero-shot reference | {zero['predicate_accuracy']:.3f} | {zero['argument1_accuracy']:.3f} | {zero['relation_argument_order_accuracy']:.3f} | {zero['truth_status_accuracy']:.3f} | {zero['exact_signed_fact_accuracy']:.3f} | {zero['exact_scene_accuracy']:.3f} | reference |",
        f"| Frozen readout (mean) | {frozen_summary['mean']['predicate_accuracy']:.3f} | {frozen_summary['mean']['argument1_accuracy']:.3f} | {frozen_summary['mean']['relation_argument_order_accuracy']:.3f} | {frozen_summary['mean']['truth_status_accuracy']:.3f} | {frozen_summary['mean']['exact_signed_fact_accuracy']:.3f} | {frozen_summary['mean']['exact_scene_accuracy']:.3f} | {'yes' if frozen_summary['passed'] else 'no'} |",
        f"| LoRA readout (mean) | {lora_summary['mean']['predicate_accuracy']:.3f} | {lora_summary['mean']['argument1_accuracy']:.3f} | {lora_summary['mean']['relation_argument_order_accuracy']:.3f} | {lora_summary['mean']['truth_status_accuracy']:.3f} | {lora_summary['mean']['exact_signed_fact_accuracy']:.3f} | {lora_summary['mean']['exact_scene_accuracy']:.3f} | {'yes' if lora_summary['passed'] else 'no'} |",
        "", "## Attribution", "",
        f"LoRA minus frozen exact-fact delta: {fact_delta:+.3f}.",
        f"LoRA minus frozen exact-scene delta: {scene_delta:+.3f}.",
        f"Family-bootstrap 95% interval for exact-fact delta: [{paired['bootstrap_95_interval'][0]:+.3f}, {paired['bootstrap_95_interval'][1]:+.3f}].",
        f"Material LoRA advantage: `{str(material).lower()}`.", "",
        "## Decision integrity", "",
        f"One V28 replay authorized: `{str(selected is not None).lower()}`.",
        "Seed selection: `none`.", "Checkpoint or hyperparameter selection: `none`.",
        f"Post-result integrity audit: `{'pass' if audit['passed'] else 'fail'}`.", "",
    ]
    (PROJECT_ROOT / args.markdown).write_text("\n".join(lines))
    print(json.dumps(audit, indent=2, sort_keys=True))
    if not audit["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
