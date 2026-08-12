# V10 results: current-state polarity decomposition

## Verdict

V10 is a clean negative transfer result. The deterministic decomposition remains exact, and the frozen 0.8B model can solve the complete task in the context-disjoint fold. None of the three readouts robustly transfers current-state polarity across held-out language families, however. The preregistered NLI-final primary fails every hard-gate family and is not eligible for LoRA or final evaluation.

The locked decision is to run a separately preregistered larger-frozen-model capacity diagnostic using the same corpus and targets. V10 itself ran no larger model and no adapter.

## Corpus and pre-model audit

The locked corpus contains 3,240 records from 90 semantic contexts, six mechanics, six language families, three state lexicons, two operator families, and 540 intervention groups. It contains 7,380 current-state hypothesis pairs and 1,620 unresolved pairs.

Generation produced zero structural, span, hypothesis, relation, allowed-value derivation, symbolic, balance, duplicate, or split-overlap errors. Current active/inactive targets are exactly balanced inside every split-by-mechanic-by-template-by-lexicon cell.

All pre-model shortcut gates passed:

- Metadata Match Maximum Fold Balanced Accuracy: 0.531 (maximum 0.600).
- Position Match Maximum Fold Balanced Accuracy: 0.531 (maximum 0.600).
- Metadata Polarity Maximum Fold Balanced Accuracy: 0.526 (maximum 0.550).
- Hypothesis Position Relation Maximum Fold Balanced Accuracy: 0.500 (maximum 0.550).

Report-only character baselines confirmed legitimate language signal: match balanced accuracy 0.884, current polarity accuracy 0.782, hypothesis-relation balanced accuracy 0.698, and temporal accuracy 1.000.

## Frozen extraction

The one authorized `mlx-community/Qwen3.5-0.8B-4bit` extraction encoded 3,492 unique determinant/evidence prompts and 6,984 unique hypothesis-conditioned prompts, covering 63,000 candidate pairs. Base prompts used 110–169 tokens, NLI prompts used 107–138, evidence spans used 11–33, and no prompt was truncated.

The representation comparison used the same locked 24 folds. `nli_final` was primary; the two direct heads were diagnostics and could not replace it after observing results.

| Representation | Context oracle polarity | Minimum fold oracle polarity | Minimum fold temporal | Minimum full allowed values | Minimum symbolic BA | Minimum flip pairs | Minimum complete groups |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `evidence_span_direct` | 1.000 | 0.170 | 0.102 | 0.186 | 0.500 | 0.000 | 0.000 |
| `mean_direct` | 0.991 | 0.477 | 0.300 | 0.269 | 0.487 | 0.020 | 0.000 |
| `nli_final` | 1.000 | 0.000 | 0.102 | 0.167 | 0.497 | 0.000 | 0.000 |

## Primary NLI-final folds

| Fold | Span | Temporal | Oracle polarity | Pair consistency | Full allowed values | Symbolic BA | Flip pairs | Complete groups |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Combined / Binary Partition / Canonical | 0.652 | 0.886 | 0.083 | 0.110 | 0.220 | 0.569 | 0.156 | 0.027 |
| Combined / Binary Partition / Entity Renamed | 0.773 | 0.817 | 0.014 | 0.071 | 0.167 | 0.498 | 0.042 | 0.000 |
| Combined / Binary Partition / Paraphrased | 0.628 | 0.866 | 0.249 | 0.290 | 0.276 | 0.576 | 0.073 | 0.000 |
| Combined / Multiway Partition / Canonical | 0.718 | 0.908 | 0.258 | 0.335 | 0.343 | 0.541 | 0.073 | 0.000 |
| Combined / Multiway Partition / Entity Renamed | 0.777 | 0.801 | 0.020 | 0.026 | 0.206 | 0.500 | 0.000 | 0.000 |
| Combined / Multiway Partition / Paraphrased | 0.675 | 0.874 | 0.036 | 0.062 | 0.202 | 0.500 | 0.000 | 0.000 |
| Context | 0.999 | 0.999 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| Lexicon / Canonical | 0.746 | 0.881 | 0.121 | 0.167 | 0.280 | 0.578 | 0.049 | 0.115 |
| Lexicon / Entity Renamed | 0.788 | 0.820 | 0.088 | 0.158 | 0.226 | 0.536 | 0.039 | 0.045 |
| Lexicon / Paraphrased | 0.679 | 0.867 | 0.241 | 0.288 | 0.262 | 0.523 | 0.049 | 0.058 |
| Mechanic / Beacon Calibration | 0.990 | 0.993 | 0.194 | 0.241 | 0.326 | 0.599 | 0.148 | 0.000 |
| Mechanic / Generator Tuning | 0.984 | 0.991 | 0.403 | 0.458 | 0.542 | 0.611 | 0.222 | 0.000 |
| Mechanic / Hatch Traversal | 1.000 | 1.000 | 0.389 | 0.417 | 0.694 | 0.708 | 0.417 | 0.083 |
| Mechanic / Mirror Power Trip | 0.904 | 0.966 | 0.194 | 0.259 | 0.300 | 0.508 | 0.010 | 0.000 |
| Mechanic / Mirror Rejection | 0.970 | 0.970 | 0.333 | 0.361 | 0.479 | 0.545 | 0.104 | 0.000 |
| Mechanic / Pressure Hatch Relock | 0.652 | 0.857 | 0.176 | 0.222 | 0.255 | 0.556 | 0.097 | 0.000 |
| Operator / Binary Partition | 0.998 | 0.998 | 0.029 | 0.056 | 0.195 | 0.497 | 0.042 | 0.000 |
| Operator / Multiway Partition | 1.000 | 1.000 | 0.198 | 0.236 | 0.353 | 0.542 | 0.099 | 0.000 |
| Template / Contrastive Correction | 0.814 | 0.102 | 0.000 | 0.000 | 0.186 | 0.500 | 0.000 | 0.000 |
| Template / Denied Claim | 0.945 | 0.964 | 0.070 | 0.070 | 0.238 | 0.510 | 0.039 | 0.000 |
| Template / Direct Assertion | 0.788 | 0.648 | 0.015 | 0.886 | 0.193 | 0.593 | 0.235 | 0.000 |
| Template / Explicit Negation | 0.898 | 0.624 | 0.433 | 0.564 | 0.407 | 0.589 | 0.157 | 0.000 |
| Template / Rejected Claim | 0.917 | 0.119 | 0.088 | 0.088 | 0.186 | 0.500 | 0.000 | 0.000 |
| Template / Scoped Rejection | 0.960 | 0.195 | 0.608 | 0.655 | 0.193 | 0.500 | 0.000 | 0.000 |

Worst primary surface cells:

- span: 0.458 at `mechanic:pressure_hatch_relock / paraphrased`.
- temporal: 0.079 at `template:contrastive_correction / entity_renamed`.
- oracle polarity: 0.000 at `mechanic:beacon_calibration / paraphrased`.
- pair consistency: 0.000 at `mechanic:beacon_calibration / paraphrased`.
- allowed values: 0.167 at `mechanic:beacon_calibration / paraphrased`.
- symbolic BA: 0.497 at `operator:binary_partition / entity_renamed`.

## Interpretation

The context fold is the crucial control: NLI-final reaches 1.000 oracle polarity, pair consistency, allowed-value accuracy, symbolic balanced accuracy, and flip-pair accuracy. The pipeline and labels are therefore internally learnable. Its collapse under held-out templates is a transfer failure rather than a broken symbolic rule or impossible target.

The direct diagnostics localize the same issue from another angle. Mean-direct retains stronger temporal transfer than evidence-span pooling, while evidence-span-direct reaches perfect context polarity. Yet their minimum held-out-template oracle polarity is only 0.477 and 0.170 respectively. A direct head learns whether the mentioned phrase correlates with a state inside known constructions, but does not reliably invert that meaning when assertion and negation operators change.

The hypothesis-conditioned final-token interface does not fix this at 0.8B layer 6. It overfits the observed construction families: minimum oracle polarity and pair consistency are both 0.000, even with the gold evidence span and gold temporal status. Span and temporal cascades make the fully predicted ledger worse, but they are not the root cause of the primary polarity failure.

The symbolic evaluator remains exact on all 3,240 records. Its robustness cannot rescue a grounding pipeline that frequently emits unresolved relation pairs: worst-fold symbolic balanced accuracy falls to approximately chance.

## Gate decision

The primary failed 14 of 14 hard checks. The locked decision is `authorize_separately_locked_larger_frozen_capacity_diagnostic`.

Next, use a separate preregistration to compare the identical primary prompts and folds with frozen 4B and 9B representations. Do not change the corpus, templates, gates, layer-selection rule, or head after seeing V10. If a larger frozen representation restores oracle polarity and pair consistency, only then consider whether a small linguistic LoRA is useful for cost reduction. V10 authorizes neither LoRA nor final-mechanic access.

## Reproducibility and firewall

- dataset: `0dbfba34e46c85b26be2744fd27065bf1386450653a287765a3df07e331d57cb`;
- pre-model audit: `661d333a94cee9de96e67acbb4c5997383856a093b5ff69ef4e7b7de823430e9`;
- frozen features: `0e7e796f3d1c6815f1b2f8a3c6a63c091db8a93362555005ecbaa5c1d37d59f8`;
- evaluation result: `372a4240d48e9131101e24af0946d5a15a95d80db615e220e970db9c246c3463`.

No larger frozen model, adapter, final mechanic, Tone Drift, V3 test record, prior holdout, untouched V8 mechanic, or V7 model result was accessed.
