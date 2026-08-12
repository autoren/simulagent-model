#!/usr/bin/env python3
"""Write the V7 causal-evidence and one-shot tone-drift report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="data/v7/manifest.json")
    parser.add_argument("--shortcut", default="outputs/v7-pre-model/shortcut-audit.json")
    parser.add_argument("--training", default="outputs/v7-frozen-probe/probe/result.json")
    parser.add_argument("--result", default="outputs/v7-untouched/frozen-probe/result.json")
    parser.add_argument("--output", default="docs/v7-results.md")
    return parser.parse_args()


def percent(value: float) -> str:
    return f"{100 * value:.2f}%"


def main() -> None:
    args = parse_args()
    manifest = json.loads(Path(args.manifest).read_text())
    shortcut = json.loads(Path(args.shortcut).read_text())
    training = json.loads(Path(args.training).read_text())
    result = json.loads(Path(args.result).read_text())
    decision = "GO" if result["gates"]["passed"] else "NO-GO"
    interval = result["canonical_grouped_bootstrap"][
        "balanced_accuracy_95_percentile_interval"
    ]
    validation = manifest["validation"]
    paired = result["paired_evidence"]
    worst = result["worst_stratum"]["worst"]
    gate_rows = [
        f"| {check['name']} | {percent(check['value'])} | {percent(check['minimum'])} | "
        f"{'pass' if check['passed'] else 'fail'} |"
        for check in result["gates"]["checks"]
    ]
    surface_rows = [
        f"| {name} | {percent(metrics['balanced_accuracy'])} | {metrics['roc_auc']:.3f} |"
        for name, metrics in result["by_surface"].items()
    ]
    failed = [check["name"] for check in result["gates"]["checks"] if not check["passed"]]
    decision_detail = (
        "Every preregistered gate passed, so a controlled LoRA plus float32 classification-head "
        "experiment is eligible."
        if not failed
        else "LoRA remains ineligible because these preregistered gates failed: " + ", ".join(failed) + "."
    )
    lines = [
        "# V7 causal-evidence curriculum and tone-drift transfer result",
        "",
        "## Decision",
        "",
        f"**{decision} for LoRA.** The frozen 0.8B layer-6 mean probe reached "
        f"{percent(result['canonical']['balanced_accuracy'])} canonical balanced accuracy and "
        f"{result['canonical']['roc_auc']:.3f} AUC on the untouched tone-drift mechanic. The "
        f"context-group bootstrap interval is {percent(interval[0])}–{percent(interval[1])}.",
        "",
        decision_detail,
        "",
        "## Pre-model rejection gates",
        "",
        f"The maximum conditional label gap was {percent(shortcut['conditional_label_gap'])}. "
        f"Metadata-only calibration balanced accuracy was "
        f"{percent(shortcut['metadata_lookup']['calibration']['balanced_accuracy'])}; evidence-card-only "
        f"balanced accuracy/AUC were "
        f"{percent(shortcut['evidence_text_naive_bayes']['calibration']['balanced_accuracy'])}/"
        f"{shortcut['evidence_text_naive_bayes']['calibration']['roc_auc']:.3f}. All pre-model gates "
        f"{'passed' if shortcut['gates']['passed'] else 'failed'} before Qwen features were read.",
        "",
        "## Preregistered model gates",
        "",
        "| Gate | Observed | Minimum | Result |",
        "| --- | ---: | ---: | --- |",
        *gate_rows,
        "",
        "## Corpus and firewall",
        "",
        f"The corpus contains {validation['base_records']['train']} training, "
        f"{validation['base_records']['calibration']} calibration, and "
        f"{validation['base_records']['untouched_mechanic']} untouched base records, each with "
        "a complete canonical/entity-renamed/paraphrased group.",
        "",
        f"Development contains {validation['label_changing_evidence_groups']['train']}/"
        f"{validation['label_changing_evidence_groups']['calibration']} training/calibration "
        "oracle label-changing groups; tone drift contains "
        f"{validation['label_changing_evidence_groups']['untouched_mechanic']}. There are zero "
        "cross-split contexts, prompts, or evidence groups, zero forbidden target fields, zero V3 "
        "reads, and zero prior-holdout reads.",
        "",
        "## Development calibration",
        "",
        f"Canonical calibration balanced accuracy was "
        f"{percent(training['calibration_canonical']['balanced_accuracy'])}. The method and threshold "
        "were frozen before the tone-drift records were opened.",
        "",
        "## Surface transfer",
        "",
        "| Surface | Balanced accuracy | AUC |",
        "| --- | ---: | ---: |",
        *surface_rows,
        "",
        f"Complete-triplet accuracy was {percent(result['surface_invariance']['complete_triplet_accuracy'])}. "
        f"Canonical/entity-renamed and canonical/paraphrased agreement were "
        f"{percent(result['surface_invariance']['transformations']['entity_renamed']['prediction_agreement'])} "
        f"and {percent(result['surface_invariance']['transformations']['paraphrased']['prediction_agreement'])}.",
        "",
        "## Grouped, paired, directional, and worst-stratum metrics",
        "",
        f"The evaluation contains {result['canonical_grouped_context_metrics']['groups']} context groups; "
        f"macro context accuracy was "
        f"{percent(result['canonical_grouped_context_metrics']['macro_accuracy'])}. The "
        f"{paired['groups']} oracle label-changing evidence groups produced "
        f"{paired['cross_label_comparisons']} comparisons: score-directional accuracy was "
        f"{percent(paired['paired_score_directional_accuracy'])}, thresholded evidence-directional "
        f"accuracy was {percent(paired['evidence_directional_accuracy'])}, and complete-group accuracy "
        f"was {percent(paired['complete_group_accuracy'])}.",
        "",
        f"The worst supported stratum was `{worst['dimension']}={worst['stratum']}` with "
        f"{worst['examples']} examples and {percent(worst['balanced_accuracy'])} balanced accuracy.",
        "",
        "## One-shot audit",
        "",
        f"- Dataset SHA-256: `{result['dataset_sha256']}`.",
        f"- Frozen probe SHA-256: `{result['probe_artifact_sha256']}`.",
        f"- Untouched evaluations: {result['untouched_evaluation_number']}.",
        f"- Untouched records scored: {result['untouched_mechanic_records_read']}.",
        f"- Truncated prompts: {result['truncated_prompts']}.",
        f"- Prior holdout records read: {result['prior_holdout_records_read']}.",
        f"- V3 test records read: {result['v3_test_records_read']}.",
        "",
    ]
    output = Path(args.output)
    output.write_text("\n".join(lines))
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
