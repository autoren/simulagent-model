# V15 results: operator-supported frozen full pipeline

## Verdict

V15 is a strong near-positive full-pipeline result, but its original locked decision remains a failure. Thirteen of fourteen transfer gates pass. Evidence matching, temporal classification, oracle polarity, allowed-value ledgers, symbolic balanced accuracy, and complete flip pairs all clear their worst-fold and worst-surface thresholds.

The only failed check is `minimum_fold_complete_intervention_group_accuracy`: 0.353 versus 0.50. A post-result topology audit shows that this number came from evaluating records outside the failing fold, not from poor predictions on that fold's evaluation set.

Final-mechanic access remains closed. The correct next step is an exact, separately preregistered scope-correct replay—not LoRA, another representation, or a relaxed performance threshold.

## Locked gates

| Check | Value | Required | Result |
| --- | ---: | ---: | --- |
| minimum_fold_span_accuracy | 0.756 | 0.65 | PASS |
| minimum_surface_span_accuracy | 0.690 | 0.60 | PASS |
| minimum_fold_temporal_accuracy | 0.913 | 0.70 | PASS |
| minimum_surface_temporal_accuracy | 0.800 | 0.65 | PASS |
| minimum_fold_oracle_polarity_accuracy | 0.830 | 0.70 | PASS |
| minimum_surface_oracle_polarity_accuracy | 0.798 | 0.65 | PASS |
| minimum_fold_nli_pair_consistency | 1.000 | 0.70 | PASS |
| minimum_surface_nli_pair_consistency | 1.000 | 0.65 | PASS |
| minimum_fold_allowed_values_accuracy | 0.773 | 0.65 | PASS |
| minimum_surface_allowed_values_accuracy | 0.752 | 0.60 | PASS |
| minimum_fold_symbolic_balanced_accuracy | 0.808 | 0.65 | PASS |
| minimum_surface_symbolic_balanced_accuracy | 0.800 | 0.60 | PASS |
| minimum_fold_complete_flip_pair_accuracy | 0.667 | 0.60 | PASS |
| minimum_fold_complete_intervention_group_accuracy | 0.353 | 0.50 | FAIL |

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

The same expansion occurs in all lexicon, transition-operator, and combined folds. Mechanic and surface folds already contain complete six-record groups and require no expansion. Across those 15 topologically valid transfer folds, the worst complete-group accuracy is **0.577** on `surface:current_observation`, which exceeds the unchanged 0.50 threshold.

This audit does not retroactively convert V15 into a preregistered pass. It identifies an evaluation-scope defect and authorizes only a scope-correct exact replay using the same frozen features, same saved heads or deterministic fits, same 26 folds, same metrics, and same 0.50 threshold. Folds with no complete in-mask groups must report the group metric as not applicable.

## Remaining diagnostic behavior

Supported surface transfer is strong. Direct Assertion, Explicit Negation, and Denied Claim are perfect end to end. Current Observation is the weakest complete-group surface at 0.577, while Scoped Rejection has the weakest oracle polarity at 0.830.

The three zero-shot semantic-operator diagnostics remain non-gating and behave as expected from V14: affirmative and negated-opposite polarity invert completely, while contrastive polarity transfers. They do not authorize claims about absent logical operators.

## Reproducibility and firewall

- V15 protocol lock: `fdcde43b41449833d9e29f51d9dd0dd2ca67405209f4bc4abb42f191d3067a60`;
- feature artifact: `7a1d0972a75b31dccc315dd513dc17d7fee2708b2a84624e974c001580405d39`;
- locked V15 result: `d45343e235238f97f0797f6056e2c96c6c8cbab33a3930557e1af9e0b7aef67c`;
- group-scope audit: `50eeae74f1d44a7f0427258cdc4d9bb254ff2fc9695443d29d8d08179f338aef`;
- fitted fold artifacts: 30, each containing match, temporal, and polarity heads;
- new model forward passes: 13,554; reused V14 prompt features: 1,512;
- LoRA runs, final-mechanic evaluations, and protected-data accesses: zero.
