#!/usr/bin/env python3
"""Write the reproducible V11 frozen-scale result report."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

from v10_protocol import file_sha256


ROOT = Path("outputs/v11-frozen-scale")
COMBINED_PATH = ROOT / "evaluation/result.json"
V10_PATH = Path("outputs/v10-frozen/evaluation/result.json")
OUTPUT_PATH = Path("docs/v11-results.md")
MODEL_PATHS = {
    "4B": ROOT / "evaluation/qwen35_4b/result.json",
    "9B": ROOT / "evaluation/qwen35_9b/result.json",
}


def minimum(folds: dict[str, Any], getter: Callable[[dict[str, Any]], float]) -> tuple[float, str]:
    return min((getter(value["overall"]), name) for name, value in folds.items())


def primary_row(label: str, result: dict[str, Any]) -> str:
    folds = result["representations"]["nli_final"]
    context = folds["context"]["overall"]
    span, _ = minimum(folds, lambda value: value["span_accuracy"])
    temporal, _ = minimum(folds, lambda value: value["temporal_accuracy_predicted_span"])
    polarity, _ = minimum(folds, lambda value: value["ablations"]["oracle_span_oracle_temporal"]["polarity_accuracy"])
    consistency, _ = minimum(folds, lambda value: value["ablations"]["oracle_span_oracle_temporal"]["hypothesis_pair_consistency"])
    allowed, _ = minimum(folds, lambda value: value["ablations"]["fully_predicted"]["allowed_values_accuracy"])
    symbolic, _ = minimum(folds, lambda value: value["ablations"]["fully_predicted"]["symbolic_identifiability"]["balanced_accuracy"])
    return (
        f"| {label} | {context['ablations']['oracle_span_oracle_temporal']['polarity_accuracy']:.3f} | "
        f"{span:.3f} | {temporal:.3f} | {polarity:.3f} | {consistency:.3f} | {allowed:.3f} | {symbolic:.3f} |"
    )


def main() -> None:
    combined = json.loads(COMBINED_PATH.read_text())
    v10 = json.loads(V10_PATH.read_text())
    larger = {label: json.loads(path.read_text()) for label, path in MODEL_PATHS.items()}
    metadata = {
        label: json.loads((ROOT / f"features/qwen35_{label.lower()}/metadata.json").read_text())
        for label in larger
    }
    lines = [
        "# V11 results: frozen-scale polarity capacity diagnostic",
        "",
        "## Verdict",
        "",
        "Frozen scale alone does not solve V10's construction-transfer failure. Both pinned larger backbones remain perfect on the context fold, and both now pass every span gate, but neither exposes construction-independent current-state polarity through the locked NLI-final linear readout.",
        "",
        "The minimum oracle-span/oracle-temporal polarity accuracy and hypothesis-pair consistency are 0.000 for 4B and 9B, exactly as in 0.8B. The result is not monotonic evidence for a capacity threshold: 4B improves selected template cells, while 9B regresses on several of them. LoRA and final-mechanic access remain closed.",
        "",
        "The locked next decision is to test a frozen joint/nonlinear token-aware relation readout before adapting model weights.",
        "",
        "## Locked comparison",
        "",
        "V11 reused V10's 3,240 records, exact target arrays, 3,492 base prompts, 6,984 NLI prompts, three readouts, 24 folds, four oracle ablations, linear-head settings, gates, and deterministic symbolic evaluator. Only the frozen backbone and the preregistered homologous depth changed.",
        "",
        "V10 layer 6 of 24 fixed the relative depth at 25%. V11 therefore used layer 8 of 32 for both larger checkpoints. The exact 4-bit model revisions were pinned before access, and no alternative layer was extracted.",
        "",
        "| Backbone | Context oracle polarity | Minimum span | Minimum temporal | Minimum oracle polarity | Minimum pair consistency | Minimum full allowed values | Minimum symbolic BA |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        primary_row("0.8B, layer 6/24", v10),
        primary_row("4B, layer 8/32", larger["4B"]),
        primary_row("9B, layer 8/32", larger["9B"]),
        "",
        "Both larger models pass the aggregate and surface span gates:",
        "",
        f"- 4B minimum fold/surface span: {larger['4B']['primary_gates']['checks'][0]['value']:.3f} / {larger['4B']['primary_gates']['checks'][1]['value']:.3f};",
        f"- 9B minimum fold/surface span: {larger['9B']['primary_gates']['checks'][0]['value']:.3f} / {larger['9B']['primary_gates']['checks'][1]['value']:.3f}.",
        "",
        "They fail the other twelve gates, including temporal transfer, oracle polarity, pair consistency, allowed ledgers, symbolic balanced accuracy, flip pairs, and complete intervention groups.",
        "",
        "## Held-out language families",
        "",
        "| Template | 0.8B temporal | 4B temporal | 9B temporal | 0.8B oracle polarity | 4B oracle polarity | 9B oracle polarity |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    template_names = [
        name for name, value in v10["representations"]["nli_final"].items() if value["kind"] == "template"
    ]
    for name in template_names:
        cells = [
            result["representations"]["nli_final"][name]["overall"]
            for result in (v10, larger["4B"], larger["9B"])
        ]
        lines.append(
            f"| {name.split(':', 1)[1].replace('_', ' ').title()} | "
            f"{cells[0]['temporal_accuracy_predicted_span']:.3f} | {cells[1]['temporal_accuracy_predicted_span']:.3f} | {cells[2]['temporal_accuracy_predicted_span']:.3f} | "
            f"{cells[0]['ablations']['oracle_span_oracle_temporal']['polarity_accuracy']:.3f} | "
            f"{cells[1]['ablations']['oracle_span_oracle_temporal']['polarity_accuracy']:.3f} | "
            f"{cells[2]['ablations']['oracle_span_oracle_temporal']['polarity_accuracy']:.3f} |"
        )

    lines.extend([
        "",
        "The direct diagnostics do not reveal a hidden monotonic scale effect either:",
        "",
        "| Backbone | Mean-direct minimum oracle polarity | Span-direct minimum oracle polarity |",
        "| --- | ---: | ---: |",
    ])
    for label, result in (("0.8B", v10), ("4B", larger["4B"]), ("9B", larger["9B"])):
        mean_value, _ = minimum(result["representations"]["mean_direct"], lambda value: value["ablations"]["oracle_span_oracle_temporal"]["polarity_accuracy"])
        span_value, _ = minimum(result["representations"]["evidence_span_direct"], lambda value: value["ablations"]["oracle_span_oracle_temporal"]["polarity_accuracy"])
        lines.append(f"| {label} | {mean_value:.3f} | {span_value:.3f} |")

    lines.extend([
        "",
        "## Interpretation",
        "",
        "Scale clearly improves determinant/evidence matching: the worst primary surface span rises from 0.458 in V10 to 0.803 at 4B and 0.718 at 9B. That validates the comparison's ability to detect a genuine capacity-related change.",
        "",
        "The polarity result is different. Even with the correct evidence and temporal status, each larger model has at least one held-out family for which every current-state decision is wrong or unresolved. Direct Assertion remains nearly perfectly pair-consistent while oriented in the wrong direction, and Contrastive Correction remains completely unresolved. The independent-hypothesis linear head is therefore the leading bottleneck after scale is controlled.",
        "",
        "Temporal classification also remains construction-bound. Contrastive Correction, Rejected Claim, and—in 9B—Scoped Rejection are often treated as non-current despite gold spans. This is a separate operator-transfer problem, but it cannot explain the oracle-polarity failure.",
        "",
        "The appropriate next diagnostic should reuse the saved frozen features and evaluate the active/inactive hypothesis pair jointly. A fixed joint linear head followed by a small preregistered nonlinear head can test whether relation information is present but not independently linearly separable. Token-span interaction features should be extracted only if both joint heads fail. This sequence is cheaper and more diagnostic than LoRA.",
        "",
        "## Decision and firewall",
        "",
        f"Combined decision: `{combined['decision']}`.",
        "",
        "No adapter, alternate layer, final mechanic, Tone Drift, V3 test record, prior holdout, untouched V8 mechanic, or V7 model result was accessed. V11 authorizes neither LoRA nor final evaluation.",
        "",
        "## Reproducibility",
        "",
        f"- V11 protocol lock: `{file_sha256(Path('configs/v11-frozen-scale-lock.json'))}`;",
        f"- 4B features: `{metadata['4B']['feature_artifact_sha256']}`;",
        f"- 9B features: `{metadata['9B']['feature_artifact_sha256']}`;",
        f"- 4B result: `{file_sha256(MODEL_PATHS['4B'])}`;",
        f"- 9B result: `{file_sha256(MODEL_PATHS['9B'])}`;",
        f"- combined result: `{file_sha256(COMBINED_PATH)}`.",
    ])
    OUTPUT_PATH.write_text("\n".join(lines) + "\n")
    print(f"wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
