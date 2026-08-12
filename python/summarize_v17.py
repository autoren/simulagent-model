#!/usr/bin/env python3
"""Write the permanent V17 one-shot final-mechanic report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Callable

from v10_protocol import file_sha256


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="data/v17-final/manifest.json")
    parser.add_argument("--features", default="outputs/v17-final/features/metadata.json")
    parser.add_argument("--result", default="outputs/v17-final/evaluation/result.json")
    parser.add_argument("--output", default="docs/v17-results.md")
    return parser.parse_args()


def minimum_location(
    result: dict[str, Any],
    selector: Callable[[dict[str, Any]], float],
    cells: bool = False,
) -> tuple[str, float]:
    values = []
    for template, payload in result["template_folds"].items():
        if cells:
            values.extend((f"{template}/{lexicon}", selector(cell)) for lexicon, cell in payload["by_surface"].items())
        else:
            values.append((template, selector(payload["overall"])))
    return min(values, key=lambda value: value[1])


def main() -> None:
    args = parse_args()
    manifest_path = Path(args.manifest)
    feature_path = Path(args.features)
    result_path = Path(args.result)
    manifest = json.loads(manifest_path.read_text())
    features = json.loads(feature_path.read_text())
    result = json.loads(result_path.read_text())
    if result["dataset_sha256"] != manifest["dataset_sha256"] or result["feature_artifact_sha256"] != features["feature_artifact_sha256"]:
        raise RuntimeError("V17 report inputs do not share the sealed dataset/features")

    passed = result["final_gates"]["passed"]
    verdict = (
        "The one-shot final mechanic passes every preregistered gate."
        if passed else "The one-shot final mechanic fails at least one preregistered gate."
    )
    claim = (
        "This supports transfer to one unseen simulator action under supported state concepts, temporal language, semantic operators, and lexicons. It does not establish arbitrary-ontology transfer and does not authorize LoRA."
        if passed else "The frozen method does not earn the preregistered final-generalization claim. V17 is now exposed diagnostic data and cannot be rescored as a final holdout."
    )
    checks = result["final_gates"]["checks"]
    lines = [
        "# V17r2 results: one-shot final-mechanic evaluation",
        "",
        verdict,
        "",
        claim,
        "",
        f"Decision: `{result['decision']}`.",
        "",
        "## Locked gates",
        "",
        "| Gate | Value | Minimum | Result |",
        "|---|---:|---:|---|",
    ]
    lines.extend(
        f"| {value['name']} | {value['value']:.3f} | {value['minimum']:.3f} | {'pass' if value['passed'] else 'fail'} |"
        for value in checks
    )
    overall = result["overall"]
    fully = overall["ablations"]["fully_predicted"]
    lines.extend([
        "",
        "## Overall final result",
        "",
        f"The final set contains {manifest['validation']['records']:,} records in {manifest['validation']['intervention_groups']} complete intervention groups. It is exactly balanced: {manifest['validation']['ambiguous_records']} ambiguous and {manifest['validation']['identifiable_records']} identifiable records.",
        "",
        f"- span accuracy: {overall['span_accuracy']:.3f};",
        f"- predicted-span temporal accuracy: {overall['temporal_accuracy_predicted_span']:.3f};",
        f"- oracle-span/oracle-temporal polarity accuracy: {overall['ablations']['oracle_span_oracle_temporal']['polarity_accuracy']:.3f};",
        f"- fully predicted allowed-values accuracy: {fully['allowed_values_accuracy']:.3f};",
        f"- fully predicted symbolic balanced accuracy: {fully['symbolic_identifiability']['balanced_accuracy']:.3f};",
        f"- complete label-flip-pair accuracy: {fully['complete_flip_pair_accuracy']:.3f}; and",
        f"- complete six-record intervention-group accuracy: {fully['complete_intervention_group_accuracy']:.3f}.",
        "",
        "## Worst transfer cells",
        "",
    ])
    metrics = [
        ("span", lambda cell: cell["span_accuracy"]),
        ("temporal", lambda cell: cell["temporal_accuracy_predicted_span"]),
        ("polarity", lambda cell: cell["ablations"]["oracle_span_oracle_temporal"]["polarity_accuracy"]),
        ("allowed values", lambda cell: cell["ablations"]["fully_predicted"]["allowed_values_accuracy"]),
        ("symbolic balanced accuracy", lambda cell: cell["ablations"]["fully_predicted"]["symbolic_identifiability"]["balanced_accuracy"]),
    ]
    for name, selector in metrics:
        location, value = minimum_location(result, selector, cells=True)
        lines.append(f"- worst template/lexicon {name}: `{location}` at {value:.3f};")
    lines.extend([
        "",
        "## Reproducibility and firewall",
        "",
        "The initial V17 constructor aborted before writing data because its normalized transition identity collapsed a read-only action. V17r2 was freshly locked before any record existed and adds the visible returned action observation to that identity; no model result informed the correction.",
        "",
        f"- construction lock: `{manifest['construction_lock_sha256']}`;",
        f"- sealed dataset: `{manifest['dataset_sha256']}`;",
        f"- evaluation lock: `{result['evaluation_lock_sha256']}`;",
        f"- final feature artifact: `{result['feature_artifact_sha256']}`;",
        f"- deployment heads: `{result['head_artifact_sha256']}`;",
        f"- result: `{file_sha256(result_path)}`;",
        f"- unique final base/NLI prompts and forward passes: {features['unique_base_prompts']:,} / {features['unique_nli_prompts']:,} / {features['new_model_forward_passes']:,};",
        f"- development-only linear fits / final evaluations / adapter runs: {result['development_linear_fits']} / {result['final_evaluation_number']} / 0; and",
        "- Tone Drift, V3 test records, prior holdouts, untouched V8 mechanics, V7 outputs, alternate models, alternate layers, alternate representations, threshold changes, and final retries: zero.",
        "",
        "V17 is permanently exposed after this result. No subsequent score on these records is a final-holdout evaluation.",
    ])
    Path(args.output).write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
