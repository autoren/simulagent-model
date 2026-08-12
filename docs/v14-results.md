# V14 results: operator-supported 4B polarity baseline

## Verdict

The supervision redesign resolves V13's apparent frozen-representation failure. With semantic operator support preserved and exact local prompts deduplicated, the fixed 4B layer-8 hypothesis-mean comparator passes every locked transfer gate.

- Worst of 26 clean transfer folds: **0.821** (required 0.70).
- Worst transfer fold × state-lexicon cell: **0.808** (required 0.65).
- All nine supported surface holdouts pass; eight are perfect and Scoped Rejection is 0.821.
- All 30 heads converge without warnings and all 26 gated folds have zero exact local-pair overlap.

The locked decision is `operator_supported_surface_transfer_passes_repair_temporal_then_full_pipeline`. The next step is to repair and retest evidence/temporal grounding on the operator-supported corpus, then run the complete symbolic pipeline. LoRA remains unnecessary and unauthorized.

## What changed the conclusion

V13's Direct Assertion holdout removed the only training family in which evidence literally mentions the true hypothesis. V14 supplies three independent surfaces for each of `affirmative_gold`, `negated_opposite`, and `contrastive_both`. Holding out one surface now leaves two examples of the same logical operator in training.

V14 also fits and scores 756 unique local NLI pairs rather than treating 11,070 repeated record occurrences as independent. The context split has 702 exact local-pair overlaps and is explicitly non-gating. Every mechanic, surface, lexicon, transition-operator, and combined transfer fold has zero overlap.

This means the V13 zero did not justify more scale or a more flexible head. It exposed a mismatch between the dataset's `template` label and the actual semantic support removed by that fold.

## Lowest clean transfer folds

| Fold | Accuracy | ROC AUC | Unique pairs |
| --- | ---: | ---: | ---: |
| surface:scoped_rejection | 0.821 | 0.912 | 78 |
| combined:binary_partition:entity_renamed | 0.905 | 1.000 | 126 |
| combined:binary_partition:paraphrased | 0.937 | 0.991 | 126 |
| combined:multiway_partition:entity_renamed | 0.944 | 1.000 | 126 |
| lexicon:entity_renamed | 0.944 | 1.000 | 234 |
| lexicon:paraphrased | 0.953 | 0.992 | 234 |
| combined:multiway_partition:paraphrased | 0.968 | 0.993 | 126 |
| combined:multiway_partition:canonical | 0.976 | 0.999 | 126 |
| mechanic:pressure_hatch_relock | 0.981 | 0.998 | 162 |
| lexicon:canonical | 0.991 | 1.000 | 234 |

The weakest fold is `surface:scoped_rejection` at 0.821. Its canonical cell is 0.846 and its entity-renamed/paraphrased cells are both 0.808, so the pass is not hidden by averaging across lexicons.

## Separate zero-shot operator diagnostics

| Held-out semantic operator | Accuracy | ROC AUC | Swap complement |
| --- | ---: | ---: | ---: |
| affirmative_gold | 0.000 | 0.000 | 1.000 |
| contrastive_both | 1.000 | 1.000 | 0.991 |
| negated_opposite | 0.000 | 0.000 | 1.000 |

Affirmative and negated-opposite operator holdouts remain exactly inverted, while contrastive-both transfers perfectly. These diagnostics are valid evidence that the supervised comparator does not infer arbitrary absent logical operators. They are not surface-transfer failures and were preregistered as non-gating.

## Reproducibility and firewall

- V14 corpus lock: `c472351dfbc5e7d1e0af34093a7ee4c0e877a0cbfbc46f677d44d2469a15c167`;
- V14 dataset: `be110b757b31416b84386f604828996d2121d07214d7401ffb3ef68c9ce5dfa8`;
- V14 model lock: `f37c28c2a60c90d3864e52a3a94f1c43746a6ceb9d602abed8ddedbbf195d950`;
- frozen feature artifact: `42c2d057b869566507d76c70672fbffa66e3b5e04a40495e8fc5e714b77d56e5`;
- evaluation result: `75ea1dad99948c787c40c1f06bc0846b2ee86b9526421d3be5bd5d6e7ca4e16f`;
- fitted heads: 30; adapter runs, final-mechanic evaluations, and protected-data accesses: zero.
