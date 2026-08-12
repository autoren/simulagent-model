#!/usr/bin/env python3
"""Write the V6 shortcut-resistant corpus and mechanic-transfer report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="data/v6/manifest.json")
    parser.add_argument("--training", default="outputs/v6-frozen-probe/probe/result.json")
    parser.add_argument("--result", default="outputs/v6-mechanic-holdout/frozen-probe/result.json")
    parser.add_argument("--output", default="docs/v6-results.md")
    return parser.parse_args()


def percent(value: float) -> str:
    return f"{100 * value:.2f}%"


def main() -> None:
    args = parse_args()
    manifest = json.loads(Path(args.manifest).read_text())
    training = json.loads(Path(args.training).read_text())
    result = json.loads(Path(args.result).read_text())
    decision = "GO" if result["gates"]["passed"] else "NO-GO"
    gate_rows = [
        f"| {check['name']} | {percent(check['value'])} | {percent(check['minimum'])} | {'pass' if check['passed'] else 'fail'} |"
        for check in result["gates"]["checks"]
    ]
    surface_rows = [
        f"| {name} | {percent(metrics['balanced_accuracy'])} | {metrics['roc_auc']:.3f} |"
        for name, metrics in result["by_surface"].items()
    ]
    interval = result["canonical_grouped_bootstrap"][
        "balanced_accuracy_95_percentile_interval"
    ]
    evidence = result["evidence_contrasts"]
    validation = manifest["validation"]
    lines = [
        "# V6 shortcut-resistant corpus and mechanic-transfer result",
        "",
        "## Decision",
        "",
        f"**{decision} for LoRA.** The fixed 0.8B layer-6 mean probe reached "
        f"{percent(result['canonical']['balanced_accuracy'])} canonical balanced accuracy and "
        f"{result['canonical']['roc_auc']:.3f} AUC on the untouched mirror-rejection mechanic. "
        f"The context-group bootstrap interval is {percent(interval[0])}–{percent(interval[1])}.",
        "",
        "## Preregistered gates",
        "",
        "| Gate | Observed | Minimum | Result |",
        "| --- | ---: | ---: | --- |",
        *gate_rows,
        "",
        "## Corpus audit",
        "",
        f"The corpus contains {validation['base_records']['train']} training, "
        f"{validation['base_records']['calibration']} calibration, and "
        f"{validation['base_records']['mechanic_holdout']} holdout base records, each with three "
        "surface views. Training uses short-start relock and power-trip; mirror rejection is "
        "reserved for the one-shot mechanic holdout.",
        "",
        f"Training/calibration include {validation['evidence_intervention_groups']['train']}/"
        f"{validation['evidence_intervention_groups']['calibration']} evidence-intervention "
        "groups. The holdout includes "
        f"{validation['evidence_intervention_groups']['mechanic_holdout']} groups, of which "
        f"{validation['label_changing_evidence_groups']['mechanic_holdout']} change the binary label.",
        "",
        "Leakage audit: zero cross-split contexts, zero cross-split prompts, zero overlap with V4 "
        "development or V5 challenge prompts/scenarios, zero privileged target fields, and zero "
        "V3 test reads.",
        "",
        "## Development calibration",
        "",
        f"Canonical calibration balanced accuracy was "
        f"{percent(training['calibration_canonical']['balanced_accuracy'])}; no layer, pooling, "
        "regularization, model size, or threshold source was selected on the mechanic holdout.",
        "",
        "## Surface transfer",
        "",
        "| Surface | Balanced accuracy | AUC |",
        "| --- | ---: | ---: |",
        *surface_rows,
        "",
        f"Complete-triplet accuracy was {percent(result['surface_invariance']['complete_triplet_accuracy'])}. "
        f"Canonical/entity-renamed agreement was "
        f"{percent(result['surface_invariance']['transformations']['entity_renamed']['prediction_agreement'])}; "
        f"canonical/paraphrased agreement was "
        f"{percent(result['surface_invariance']['transformations']['paraphrased']['prediction_agreement'])}.",
        "",
        "## Evidence interventions",
        "",
        f"The label-changing holdout groups contain {evidence['cross_label_comparisons']} cross-label "
        f"comparisons. Directional accuracy was {percent(evidence['directional_accuracy'])}, and "
        f"complete-group classification was {percent(evidence['complete_group_accuracy'])}.",
        "",
        "## Firewall",
        "",
        f"- V6 dataset SHA-256: `{result['dataset_sha256']}`.",
        f"- Frozen probe SHA-256: `{result['probe_artifact_sha256']}`.",
        f"- Holdout evaluations: {result['holdout_evaluation_number']}.",
        f"- Holdout records scored: {result['mechanic_holdout_records_read']}.",
        f"- Truncated prompts: {result['truncated_prompts']}.",
        f"- V3 test records read: {result['v3_test_records_read']}.",
        "",
    ]
    Path(args.output).write_text("\n".join(lines))
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
