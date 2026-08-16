#!/usr/bin/env python3
"""Reproduce and integrity-audit the one V36 confirmation result."""

from __future__ import annotations

import argparse
import json

from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT
from v36_evaluation import decision_from_checks, gate_checks, score_confirmation


def jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", default="outputs/v36-independent-confirmation/evaluation/result.json")
    parser.add_argument("--audit", default="outputs/v36-independent-confirmation/post-result-audit.json")
    parser.add_argument("--markdown", default="docs/v36-results.md")
    args = parser.parse_args()
    result_path, audit_path, markdown_path = map(lambda value: (PROJECT_ROOT / value).resolve(), (args.result, args.audit, args.markdown))
    result = json.loads(result_path.read_text()); errors = []
    feature_lock_path = PROJECT_ROOT / result["features_lock"]
    feature_lock = json.loads(feature_lock_path.read_text())
    seal_path = PROJECT_ROOT / feature_lock["confirmation_seal"]
    seal = json.loads(seal_path.read_text())
    implementation_path = PROJECT_ROOT / seal["implementation_lock"]
    implementation = json.loads(implementation_path.read_text())
    config, v32_config = implementation["config_payload"], implementation["v32_config_payload"]
    for path, expected in implementation["implementation"].items():
        if file_sha256(PROJECT_ROOT / path) != expected:
            errors.append(f"V36 implementation changed: {path}")
    corpus_path, predictions_path = PROJECT_ROOT / seal["corpus_artifact"], PROJECT_ROOT / result["predictions"]
    if file_sha256(predictions_path) != result["predictions_sha256"]:
        errors.append("V36 predictions changed")
    rows, predictions = sorted(jsonl(corpus_path), key=lambda row: row["id"]), jsonl(predictions_path)
    metrics = score_confirmation(rows, predictions, v32_config, config["execution"]["bootstrapSeed"], config["execution"]["bootstrapReplicates"])
    checks = gate_checks(metrics, config); decision, magnitude = decision_from_checks(metrics, checks)
    comparisons = {
        "metrics": metrics == result["metrics"], "gate_checks": checks == result["gate_checks"],
        "passed": all(checks.values()) == result["passed"], "decision": decision == result["decision"],
        "confirmation_magnitude": magnitude == result["confirmation_magnitude"],
    }
    if not all(comparisons.values()):
        errors.append("V36 result does not reproduce from saved predictions")
    access = result["data_access"]
    access_checks = {
        "one_evaluation": access["confirmation_evaluations"] == 1,
        "all_records": access["confirmation_records_scored"] == 1170,
        "no_selection": access["selection_runs"] == 0, "no_threshold_changes": access["threshold_changes"] == 0,
        "no_evaluation_model_forward": access["model_forward_passes"] == 0, "no_interface_refit": access["interface_fit_runs"] == 0,
        "no_v32_evaluation": access["v32_evaluation_records_read"] == 0, "no_v28": access["v28_integration_replays"] == 0,
        "no_adapter": access["adapter_training_runs"] == 0,
    }
    if not all(access_checks.values()):
        errors.append("V36 evaluation access ledger violates lock")
    feature_metadata = json.loads((PROJECT_ROOT / feature_lock["feature_metadata"]).read_text())
    chain_checks = {
        "implementation_bound": seal["implementation_lock_sha256"] == file_sha256(implementation_path),
        "interface_frozen_before_corpus": seal["interface_lock_sha256"] == file_sha256(PROJECT_ROOT / seal["interface_lock"]),
        "corpus_sealed": seal["corpus_artifact_sha256"] == file_sha256(corpus_path),
        "exact_forward_budget": feature_metadata["backbone_forward_passes"] == 3510,
        "zero_truncation": feature_metadata["truncated_prompts"] == 0,
        "feature_lock_bound": result["features_lock_sha256"] == file_sha256(feature_lock_path),
    }
    if not all(chain_checks.values()):
        errors.append("V36 lock chain or extraction budget failed")
    audit = {
        "schema_version": 36, "experiment": "v36_post_result_audit", "result": str(result_path.relative_to(PROJECT_ROOT)), "result_sha256": file_sha256(result_path),
        "passed": not errors, "decision": "accept_v36_confirmation_result" if not errors else "reject_v36_confirmation_result", "errors": errors,
        "reproduction_checks": comparisons, "access_checks": access_checks, "chain_checks": chain_checks,
        "confirmation_passed": all(checks.values()), "scientific_decision": decision, "confirmation_magnitude": magnitude,
    }
    audit_path.parent.mkdir(parents=True, exist_ok=True); audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    m = metrics
    lines = [
        "# V36 results: independent modular-interface confirmation", "", f"Decision: `{decision}`.", "",
        "V36 is an independent development confirmation across 15 newly frozen supported-language families. It is not a natural-language population estimate or an end-to-end relational result.", "",
        "## One-shot confirmation", "", "| Metric | Result | Gate |", "|---|---:|---:|",
        f"| Predicate | {m['predicate_accuracy']:.3f} | 0.980 |", f"| Exact atom | {m['atom_exact_accuracy']:.3f} | 0.950 |",
        f"| Relation order | {m['relation_order_accuracy']:.3f} | 0.950 |", f"| Lexical sign | {m['lexical_sign_accuracy']:.3f} | 0.950 |",
        f"| Outer operation | {m['outer_operation_accuracy']:.3f} | 0.950 |", f"| Compiled truth | {m['compiled_truth_accuracy']:.3f} | 0.950 |",
        f"| Exact fact | {m['compiled_exact_fact_accuracy']:.3f} | 0.900 |", f"| Exact scene | {m['exact_scene_accuracy']:.3f} | 0.800 |",
        f"| Worst family exact fact | {m['worst_surface_family_exact_fact']:.3f} | 0.800 |", f"| Negative-composition exact fact | {m['negative_composition_exact_fact']:.3f} | 0.900 |",
        "", "## Interpretation", "", f"Confirmation magnitude: `{magnitude}`. Exact-fact change from V35 exposed calibration: {result['exact_fact_change_from_v35_development']:+.3f}.", "",
        f"All registered gates passed: `{str(all(checks.values())).lower()}`. Family-bootstrap exact-fact interval: [{m['family_bootstrap_exact_fact_95_interval'][0]:.3f}, {m['family_bootstrap_exact_fact_95_interval'][1]:.3f}].", "",
        f"Post-result integrity audit: `{'pass' if audit['passed'] else 'fail'}`.",
    ]
    markdown_path.write_text("\n".join(lines) + "\n")
    print(json.dumps(audit, indent=2, sort_keys=True))
    if not audit["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
