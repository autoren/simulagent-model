#!/usr/bin/env python3
"""Summarize V15's full-pipeline result and group-scope audit."""

from __future__ import annotations

import json
from pathlib import Path

from v10_protocol import file_sha256


RESULT = Path("outputs/v15-full-pipeline/evaluation/result.json")
FEATURES = Path("outputs/v15-full-pipeline/features/v15-full-features.npz")
AUDIT = Path("outputs/v15-full-pipeline/group-scope-audit.json")
OUTPUT = Path("docs/v15-results.md")


def main() -> None:
    result = json.loads(RESULT.read_text())
    audit = json.loads(AUDIT.read_text())
    rows = []
    for check in result["primary_transfer_gates"]["checks"]:
        rows.append(
            f"| {check['name']} | {check['value']:.3f} | {check['minimum']:.2f} | "
            f"{'PASS' if check['passed'] else 'FAIL'} |"
        )
    text = f"""# V15 results: operator-supported frozen full pipeline

## Verdict

V15 is a strong near-positive full-pipeline result, but its original locked decision remains a failure. Thirteen of fourteen transfer gates pass. Evidence matching, temporal classification, oracle polarity, allowed-value ledgers, symbolic balanced accuracy, and complete flip pairs all clear their worst-fold and worst-surface thresholds.

The only failed check is `minimum_fold_complete_intervention_group_accuracy`: 0.353 versus 0.50. A post-result topology audit shows that this number came from evaluating records outside the failing fold, not from poor predictions on that fold's evaluation set.

Final-mechanic access remains closed. The correct next step is an exact, separately preregistered scope-correct replay—not LoRA, another representation, or a relaxed performance threshold.

## Locked gates

| Check | Value | Required | Result |
| --- | ---: | ---: | --- |
{chr(10).join(rows)}

The most important clean minima are:

- span accuracy: 0.756 fold / 0.690 lexicon cell;
- temporal accuracy: 0.913 / 0.800;
- oracle polarity: 0.830 / 0.798;
- fully predicted allowed values: 0.773 / 0.752;
- symbolic balanced accuracy: 0.808 / 0.800;
- complete flip-pair accuracy: 0.667.

The repeated-local-prompt context control is exactly 1.000 end to end and was correctly excluded from gating.

## Why the group gate failed

`operator:multiway_partition` evaluates 720 entity-renamed records. On those records the full pipeline is perfect: span, temporal, polarity, allowed values, symbolic accuracy, flip pairs, and complete ledgers are all 1.000.

That evaluation mask contains zero complete intervention groups because a complete group requires all three lexicons. The inherited V10 `group_scope` function therefore expanded the group calculation to 2,160 records across canonical, entity-renamed, and paraphrased surfaces. Two thirds of those records were outside the fold's evaluation mask. The resulting 0.353 is a stronger, different experiment and should not be labeled a per-fold complete-group score.

The same expansion occurs in all lexicon, transition-operator, and combined folds. Mechanic and surface folds already contain complete six-record groups and require no expansion. Across those {len(audit['topologically_applicable_folds'])} topologically valid transfer folds, the worst complete-group accuracy is **{audit['scope_correct_minimum']['value']:.3f}** on `{audit['scope_correct_minimum']['fold']}`, which exceeds the unchanged 0.50 threshold.

This audit does not retroactively convert V15 into a preregistered pass. It identifies an evaluation-scope defect and authorizes only a scope-correct exact replay using the same frozen features, same saved heads or deterministic fits, same 26 folds, same metrics, and same 0.50 threshold. Folds with no complete in-mask groups must report the group metric as not applicable.

## Remaining diagnostic behavior

Supported surface transfer is strong. Direct Assertion, Explicit Negation, and Denied Claim are perfect end to end. Current Observation is the weakest complete-group surface at 0.577, while Scoped Rejection has the weakest oracle polarity at 0.830.

The three zero-shot semantic-operator diagnostics remain non-gating and behave as expected from V14: affirmative and negated-opposite polarity invert completely, while contrastive polarity transfers. They do not authorize claims about absent logical operators.

## Reproducibility and firewall

- V15 protocol lock: `{result['protocol_lock_sha256']}`;
- feature artifact: `{file_sha256(FEATURES)}`;
- locked V15 result: `{file_sha256(RESULT)}`;
- group-scope audit: `{file_sha256(AUDIT)}`;
- fitted fold artifacts: 30, each containing match, temporal, and polarity heads;
- new model forward passes: 13,554; reused V14 prompt features: 1,512;
- LoRA runs, final-mechanic evaluations, and protected-data accesses: zero.
"""
    OUTPUT.write_text(text)
    print(f"wrote {OUTPUT}")


if __name__ == "__main__":
    main()
