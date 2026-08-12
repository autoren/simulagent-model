#!/usr/bin/env python3
"""Summarize the V14 corpus correction and 4B polarity baseline."""

from __future__ import annotations

import json
from pathlib import Path

from v10_protocol import file_sha256


RESULT = Path("outputs/v14-4b-baseline/evaluation/result.json")
FEATURES = Path("outputs/v14-4b-baseline/features/v14-4b-token-mean-features.npz")
OUTPUT = Path("docs/v14-results.md")


def main() -> None:
    result = json.loads(RESULT.read_text())
    rows = []
    for name, value in result["primary_folds"].items():
        if name == "context":
            continue
        metric = value["overall"]
        rows.append((metric["accuracy"], name, metric["roc_auc"], metric["examples"]))
    rows.sort()
    diagnostic_rows = []
    for name, value in result["zero_shot_operator_diagnostics"].items():
        metric = value["overall"]
        diagnostic_rows.append(
            f"| {name.removeprefix('semantic_operator:')} | {metric['accuracy']:.3f} | "
            f"{metric['roc_auc']:.3f} | {metric['swap_complement_accuracy']:.3f} |"
        )
    text = f"""# V14 results: operator-supported 4B polarity baseline

## Verdict

The supervision redesign resolves V13's apparent frozen-representation failure. With semantic operator support preserved and exact local prompts deduplicated, the fixed 4B layer-8 hypothesis-mean comparator passes every locked transfer gate.

- Worst of 26 clean transfer folds: **{result['primary_transfer_gates']['checks'][0]['value']:.3f}** (required 0.70).
- Worst transfer fold × state-lexicon cell: **{result['primary_transfer_gates']['checks'][1]['value']:.3f}** (required 0.65).
- All nine supported surface holdouts pass; eight are perfect and Scoped Rejection is 0.821.
- All 30 heads converge without warnings and all 26 gated folds have zero exact local-pair overlap.

The locked decision is `{result['decision']}`. The next step is to repair and retest evidence/temporal grounding on the operator-supported corpus, then run the complete symbolic pipeline. LoRA remains unnecessary and unauthorized.

## What changed the conclusion

V13's Direct Assertion holdout removed the only training family in which evidence literally mentions the true hypothesis. V14 supplies three independent surfaces for each of `affirmative_gold`, `negated_opposite`, and `contrastive_both`. Holding out one surface now leaves two examples of the same logical operator in training.

V14 also fits and scores 756 unique local NLI pairs rather than treating 11,070 repeated record occurrences as independent. The context split has 702 exact local-pair overlaps and is explicitly non-gating. Every mechanic, surface, lexicon, transition-operator, and combined transfer fold has zero overlap.

This means the V13 zero did not justify more scale or a more flexible head. It exposed a mismatch between the dataset's `template` label and the actual semantic support removed by that fold.

## Lowest clean transfer folds

| Fold | Accuracy | ROC AUC | Unique pairs |
| --- | ---: | ---: | ---: |
{chr(10).join(f'| {name} | {accuracy:.3f} | {auc:.3f} | {examples} |' for accuracy, name, auc, examples in rows[:10])}

The weakest fold is `surface:scoped_rejection` at 0.821. Its canonical cell is 0.846 and its entity-renamed/paraphrased cells are both 0.808, so the pass is not hidden by averaging across lexicons.

## Separate zero-shot operator diagnostics

| Held-out semantic operator | Accuracy | ROC AUC | Swap complement |
| --- | ---: | ---: | ---: |
{chr(10).join(diagnostic_rows)}

Affirmative and negated-opposite operator holdouts remain exactly inverted, while contrastive-both transfers perfectly. These diagnostics are valid evidence that the supervised comparator does not infer arbitrary absent logical operators. They are not surface-transfer failures and were preregistered as non-gating.

## Reproducibility and firewall

- V14 corpus lock: `c472351dfbc5e7d1e0af34093a7ee4c0e877a0cbfbc46f677d44d2469a15c167`;
- V14 dataset: `be110b757b31416b84386f604828996d2121d07214d7401ffb3ef68c9ce5dfa8`;
- V14 model lock: `{result['protocol_lock_sha256']}`;
- frozen feature artifact: `{file_sha256(FEATURES)}`;
- evaluation result: `{file_sha256(RESULT)}`;
- fitted heads: 30; adapter runs, final-mechanic evaluations, and protected-data accesses: zero.
"""
    OUTPUT.write_text(text)
    print(f"wrote {OUTPUT}")


if __name__ == "__main__":
    main()
