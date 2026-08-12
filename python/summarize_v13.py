#!/usr/bin/env python3
"""Write the V13 token-local result and operator-support diagnosis."""

from __future__ import annotations

import json
from pathlib import Path

from v10_protocol import file_sha256


RESULT_PATH = Path("outputs/v13-token-local/evaluation/result.json")
FEATURE_PATH = Path("outputs/v13-token-local/features/v13-token-local-features.npz")
METADATA_PATH = Path("outputs/v13-token-local/features/metadata.json")
AUDIT_PATH = Path("outputs/v13-token-local/operator-support-audit.json")
OUTPUT_PATH = Path("docs/v13-results.md")
HEAD_LABELS = {
    "hypothesis_last_linear": "Last-token linear",
    "hypothesis_mean_linear": "Hypothesis-mean linear",
    "hypothesis_token_joint_mlp": "Token-joint 32-unit MLP",
}


def main() -> None:
    result = json.loads(RESULT_PATH.read_text())
    metadata = json.loads(METADATA_PATH.read_text())
    audit = json.loads(AUDIT_PATH.read_text())
    gate_rows = []
    template_rows = []
    templates = [
        "contrastive_correction", "denied_claim", "direct_assertion",
        "explicit_negation", "rejected_claim", "scoped_rejection",
    ]
    for name, values in result["heads"].items():
        checks = {item["name"]: item["value"] for item in values["gates"]["checks"]}
        context = values["folds"]["context"]["overall"]["accuracy"]
        warnings = sum(cell["convergence_warnings"] for cell in values["folds"].values())
        gate_rows.append(
            f"| {HEAD_LABELS[name]} | {context:.3f} | {checks['minimum_fold_accuracy']:.3f} | "
            f"{checks['minimum_surface_accuracy']:.3f} | {warnings} |"
        )
    for template in templates:
        fold = f"template:{template}"
        cells = []
        for name in HEAD_LABELS:
            metric = result["heads"][name]["folds"][fold]["overall"]
            cells.append(f"{metric['accuracy']:.3f} / {metric['roc_auc']:.3f}")
        template_rows.append(f"| {template.replace('_', ' ').title()} | " + " | ".join(cells) + " |")

    support_rows = []
    for template in templates:
        counts = audit["by_template"][template]
        holdout = audit["template_holdouts"][template]
        unsupported = ", ".join(holdout["unsupported_evaluation_signatures"]) or "none"
        support_rows.append(
            f"| {template.replace('_', ' ').title()} | {counts['gold_only']} | "
            f"{counts['opposite_only']} | {counts['both']} | {unsupported} |"
        )

    text = f"""# V13 results: 4B token-local relation diagnostic

## Verdict

Token-local pooling substantially improves construction transfer, but it does not satisfy the locked worst-family gate. All three heads are perfect on the context fold and most held-out construction families; all three are also exactly reversed on held-out Direct Assertion, with 0.000 accuracy and 0.000 ROC AUC.

The decision is `{result['decision']}`. Frozen feature probing stops here. The next phase should redesign supervision and evaluation around semantic-operator support rather than try another model scale, layer, pooling rule, or post-hoc classifier.

This result is more informative than a generic negative. The hypothesis-mean linear head reaches at least 0.751 on every non-Direct fold and at least 0.977 on five of six held-out template families. The single 0.000 fold is explained by a structural support gap in the current template taxonomy.

## Locked V13 results

| Head | Context accuracy | Worst fold accuracy | Worst surface accuracy | Convergence warnings |
| --- | ---: | ---: | ---: | ---: |
{chr(10).join(gate_rows)}

Required minima were 0.70 in every fold and 0.65 in every fold-by-surface cell. None passes because Direct Assertion is exactly inverted.

| Held-out template | Last linear acc / AUC | Mean linear acc / AUC | Joint MLP acc / AUC |
| --- | ---: | ---: | ---: |
{chr(10).join(template_rows)}

The hypothesis-mean representation is the strongest frozen polarity representation tested so far. It solves Contrastive Correction, Denied Claim, Explicit Negation, Rejected Claim, and nearly all Scoped Rejection examples at 4B. The MLP adds no robust benefit and the last token is weaker on Scoped Rejection.

## Operator-support audit

For each current determinant, the audit asks whether the gold evidence literally contains the gold hypothesis, the opposite hypothesis, or both. Counts cover all {audit['current_determinants']:,} current determinants.

| Template | Gold only | Opposite only | Both | Signature absent when held out |
| --- | ---: | ---: | ---: | --- |
{chr(10).join(support_rows)}

Direct Assertion is the only `gold_only` family. When it is held out, training contains zero examples in which the literally mentioned hypothesis is the true one: four remaining families mention the opposite hypothesis under rejection/negation, and Contrastive Correction mentions both. Likewise, Contrastive Correction is the only `both` family, although the token-local representation happens to generalize to it.

Therefore the current `template` fold conflates two tests:

1. surface-form transfer within a known logical operator; and
2. zero-shot transfer to an entirely absent semantic mention-orientation operator.

The Direct Assertion zero is valid evidence that the supervised head did not perform zero-shot logical transfer. It is not evidence that a broader classifier architecture would fix the issue, and it should not be treated as an ordinary paraphrase failure.

## Correct V14 direction

V14 should rebuild the linguistic taxonomy before any additional model training:

- define semantic operator signatures such as affirmative assertion (`gold_only`), negated/rejected opposite (`opposite_only`), and contrastive correction (`both`);
- create at least two independently worded surface families per operator signature;
- use primary folds that hold out a surface family while retaining the same operator signature in training;
- report a separate, explicitly labeled zero-shot operator-holdout benchmark, never mixing it into the paraphrase-transfer gate;
- balance temporal operators independently so temporal holdouts have analogous support;
- retain the exact symbolic evaluator and protected-data firewall.

The strongest fixed baseline for that redesigned corpus should be 4B layer-8 hypothesis-mean signed comparison. Only after it passes supported surface holdouts should a separately locked adapter objective be considered. LoRA is not authorized by V13 failure.

## Reproducibility

- V13 protocol lock: `{result['protocol_lock_sha256']}`;
- token-local feature artifact: `{file_sha256(FEATURE_PATH)}`;
- V13 result: `{file_sha256(RESULT_PATH)}`;
- operator-support audit: `{file_sha256(AUDIT_PATH)}`;
- prompt count/hash: {metadata['unique_nli_prompts']:,} / `{metadata['nli_prompt_text_sha256']}`;
- fitted heads: 72 (three heads × 24 folds);
- convergence warnings, truncations, adapter runs, and protected-data accesses: zero.
"""
    OUTPUT_PATH.write_text(text)
    print(f"wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
