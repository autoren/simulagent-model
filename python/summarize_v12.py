#!/usr/bin/env python3
"""Write the V12 joint-readout result summary."""

from __future__ import annotations

import json
from pathlib import Path

from v10_protocol import file_sha256


ROOT = Path("outputs/v12-joint-readout/evaluation")
RESULT = ROOT / "result.json"
OUTPUT = Path("docs/v12-results.md")
LABELS = {
    "qwen35_0_8b": "0.8B",
    "qwen35_4b": "4B",
    "qwen35_9b": "9B",
}
HEADS = {
    "signed_difference_linear": "Signed-difference linear",
    "joint_mlp": "Joint 32-unit MLP",
}


def value(result, head, model, fold, field):
    return result["heads"][head][model]["folds"][fold]["overall"][field]


def main() -> None:
    result = json.loads(RESULT.read_text())
    if result["schema_version"] != 12:
        raise RuntimeError("unexpected V12 schema")
    rows = []
    for head, head_label in HEADS.items():
        for model in result["model_order"]:
            values = result["heads"][head][model]
            folds = values["folds"]
            gate_values = {item["name"]: item["value"] for item in values["gates"]["checks"]}
            rows.append(
                f"| {head_label} | {LABELS[model]} | "
                f"{folds['context']['overall']['accuracy']:.3f} | "
                f"{gate_values['minimum_fold_accuracy']:.3f} | "
                f"{gate_values['minimum_surface_accuracy']:.3f} | "
                f"{sum(cell['convergence_warnings'] for cell in folds.values())} |"
            )

    template_rows = []
    templates = [
        "contrastive_correction", "denied_claim", "direct_assertion",
        "explicit_negation", "rejected_claim", "scoped_rejection",
    ]
    for template in templates:
        fold = f"template:{template}"
        template_rows.append(
            f"| {template.replace('_', ' ').title()} | "
            + " | ".join(
                f"{value(result, head, model, fold, 'accuracy'):.3f}"
                for head in HEADS for model in result["model_order"]
            )
            + " |"
        )

    direct_diagnostics = []
    for head, head_label in HEADS.items():
        for model in result["model_order"]:
            fold = result["heads"][head][model]["folds"]["template:direct_assertion"]["overall"]
            direct_diagnostics.append(
                f"| {head_label} | {LABELS[model]} | {fold['accuracy']:.3f} | "
                f"{fold['roc_auc']:.3f} | {fold['swap_complement_accuracy']:.3f} |"
            )

    text = f"""# V12 results: frozen joint-hypothesis readout

## Verdict

Jointly comparing the active and inactive hypotheses does not repair construction transfer at any tested scale. The signed-difference linear comparator fails the locked gate for 0.8B, 4B, and 9B, so the preregistered joint MLP ran; it also fails for all three.

This is a clean representation-level negative, not an optimization failure. Every model reaches 1.000 context-fold accuracy, no head emits a convergence warning, and the held-out Direct Assertion family is nearly or exactly perfectly rank-reversed. The final-token hypothesis representations contain a stable distinction, but its orientation remains tied to the language families present during training.

The locked decision is `{result['decision']}`. The next permitted experiment is one token/span-local extraction at the smallest backbone that already passed V11's span gates (4B). LoRA and final-mechanic access remain closed.

## Gate results

V12 used 7,380 gold-matched, gold-current determinant examples, exactly balanced between active and inactive. It reused all 24 locked V10 folds and did no new model inference before the result below.

| Head | Backbone | Context accuracy | Worst fold accuracy | Worst surface accuracy | Convergence warnings |
| --- | --- | ---: | ---: | ---: | ---: |
{chr(10).join(rows)}

The required minima were 0.70 for every fold and 0.65 for every non-empty fold-by-surface cell. No head/backbone combination passes.

## Held-out construction families

| Held-out template | Linear 0.8B | Linear 4B | Linear 9B | MLP 0.8B | MLP 4B | MLP 9B |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
{chr(10).join(template_rows)}

The result is highly structured rather than uniformly noisy. Denied Claim, Rejected Claim, and Scoped Rejection often transfer well, while Direct Assertion is systematically inverted and Contrastive Correction remains weak. Increasing scale does not remove that split.

## Why the zeroes are informative

| Head | Backbone | Direct Assertion accuracy | ROC AUC | Swap-complement accuracy |
| --- | --- | ---: | ---: | ---: |
{chr(10).join(direct_diagnostics)}

An ROC AUC of 0.000 means the held-out active/inactive ordering is perfectly reversed, not absent. For 9B, both heads also have 1.000 swap-complement accuracy on Direct Assertion: swapping the two hypothesis vectors flips the prediction exactly, but the learned orientation is wrong for every example. A larger or nonlinear final-token head therefore cannot by itself supply the missing construction-independent semantics.

The MLP result also narrows the next step. Its input contained both the hypothesis-pair mean and signed difference, so it could condition the relative decision on common evidence/context information. Its failure rules out a simple missing mean-by-difference interaction at the final token.

## Next experiment

The next representation should stay frozen and use 4B only. It should preserve the locked layer and prompts, but extract the last contextualized token inside each hypothesis rather than the generic assistant-generation token. In this causal prompt order, that token has attended to the evidence and the complete hypothesis and is the narrowest token-local relation representation available without changing the language task. A signed pair comparison should remain primary; one fixed nonlinear head may be conditional.

Temporal operator transfer remains a separate unresolved component. Even a successful token-local polarity readout must be combined with a repaired temporal head and re-run through the full locked symbolic pipeline before any final-mechanic access.

## Reproducibility and firewall

- V12 protocol lock: `{result['protocol_lock_sha256']}`;
- V12 result: `{file_sha256(RESULT)}`;
- fitted head artifacts: 144 (two heads × three models × 24 folds);
- new feature extractions: {result['new_feature_extractions']};
- adapter runs and final-mechanic evaluations: zero.

All protected-access counters remain zero. V12 authorizes neither LoRA nor final evaluation.
"""
    OUTPUT.write_text(text)
    print(f"wrote {OUTPUT}")


if __name__ == "__main__":
    main()
