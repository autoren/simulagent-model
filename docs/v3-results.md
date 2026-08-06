# Dataset V3 and outcome-count calibration results

## Decision

The V3 data revision passed its split gates, but the preregistered three-seed Qwen3.5-0.8B
calibration gate failed. Oracle-count transition generation was therefore not run.

This is a sharper negative result than V2: removing the ambiguity-rate shift did not remove the
adapter's all-one/all-two operating modes. A simple visible-input token classifier performs well,
however, so the remaining failure is in the current language-model objective, optimization, or
calibration—not an absence of learnable signal in the prompt.

## V3 dataset gate

V3 assigns all actions from one observation-state context to the same split and optimizes the
group assignment over identifiability, exact outcome count, action family, scenario family, and
mechanic tags with at least 100 records of support.

| Check | Result |
| --- | ---: |
| Records | 1,525 |
| Train / validation / test | 1,218 / 154 / 153 |
| Context groups | 154 / 19 / 19 |
| Ambiguity rate | 39.98% / 40.91% / 40.52% |
| Maximum ambiguity-rate gap | 0.93 points |
| Maximum common-mechanic share gap | 5.54 points |
| Exact prompt overlaps | 0 |
| Observation-context overlaps | 0 |
| Dataset SHA-256 | `fe62d8bb3877792d301bb6907abc2bb52df7cd2ccce3f6ffcfc9c70b7f1ed6e3` |

The current test set remains diagnostic rather than a fresh blind holdout because prior
experiments informed this methodology. V3 model and baseline selection used validation only.

## Validation baselines

| Baseline | Exact count | Macro count | Ambiguous exact | Balanced ID | Ambiguity F1 | MAE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Always one | 59.09% | 20.00% | 0.00% | 50.00% | 0.00% | 0.545 |
| Action majority | 59.09% | 20.00% | 0.00% | 50.00% | 0.00% | 0.545 |
| Nearest neighbour | 38.96% | 16.53% | 26.98% | 38.71% | 29.23% | 0.786 |
| Balanced-prior token Naive Bayes | 74.03% | 31.55% | 52.38% | 73.08% | 66.06% | 0.377 |

Token Naive Bayes reads only the serialized agent input. It predicts count 1 or the most common
ambiguous training count (2); consequently it has zero exact accuracy on counts 3 through 5 and
only 31.55% macro count accuracy. Its strong binary performance still establishes that visible
lexical/statistical features carry substantial identifiability signal.

## Qwen3.5-0.8B run

Each seed trained 3.608M LoRA parameters in the final 16 transformer layers for 400 updates at a
`1e-5` learning rate. Checkpoints were saved every 100 updates. Training and loss evaluation used
the full 154-record validation split; generated evaluation constrained the next-token decision to
ASCII digits 1 through 5. Peak unified memory was 6.28 GB. No test predictions were generated.

Validation selection used balanced identifiability first, ambiguity F1 second, macro count third,
and the earlier step as the final tie-breaker.

| Seed | Selected step | Exact count | Balanced ID | Ambiguity F1 | Prediction distribution |
| ---: | ---: | ---: | ---: | ---: | --- |
| 0 | 100 | 59.09% | 50.00% | 0.00% | `1:154` |
| 1 | 100 | 59.09% | 50.00% | 0.00% | `1:154` |
| 2 | 100 | 59.09% | 50.00% | 0.00% | `1:154` |

Seeds 0 and 2 predicted count 1 for every validation prompt at every checkpoint. Seed 1 did the
same through step 300, then flipped at step 400 to `1:11, 2:143`. That late checkpoint reached
49.33% balanced ID and 56.31% ambiguity F1; the F1 came from 92.06% ambiguity recall combined
with only 40.56% precision, not context-sensitive classification.

Teacher-forced full-validation losses did not expose the output modes. At step 100 all three seeds
had loss 0.171. Final losses were 0.150, 0.172, and 0.164 for seeds 0, 1, and 2 respectively, while
their admissible generated selections were equivalent constant predictors.

## Calibration gate

The gate was fixed before training:

1. Every selected seed must exceed 50% balanced identifiability.
2. At least two of three seeds and their mean must reach 55%.
3. The selected-seed range must be no more than ten points.
4. Every selected checkpoint must predict both identifiable and ambiguous cases.

Only the range condition passed. Mean selected balanced identifiability was exactly 50.00%, and
all selected checkpoints were constant. The overall gate failed, so the oracle-count generator
stage was deliberately skipped.

## Ineligible logit diagnostic

The default argmax masks some weak ranking information. The best validation ROC AUC observed per
seed was 0.550 (seed 0, step 400), 0.563 (seed 1, step 400), and 0.560 (seed 2, step 200).
Thresholds fitted post hoc on those same validation labels reached 59.52%, 56.47%, and 55.98%
balanced ID. These values are optimistic and were not used for selection or the gate.

The signal is weak and unstable across checkpoints, but it suggests that the adapter sometimes
orders examples better than its globally biased digit decision reveals.

## Interpretation

V3 rules out the largest V2 confound: train, validation, and test no longer reward different
majority classes. The output oscillation nevertheless persists. At the same time, token Naive
Bayes substantially exceeds every LoRA checkpoint on validation. The next bottleneck is therefore
the five-way one-token SFT formulation and its global token prior, not split balance or model size.

The next controlled experiment should be:

1. Reduce stage 1 to binary `identifiable` versus `ambiguous` classification.
2. Reserve context groups from the current training partition as a calibration fold.
3. Train a balanced classifier objective or classification head rather than free next-token SFT.
4. Fit exactly one decision threshold on the calibration fold and evaluate it once on V3
   validation across multiple seeds.
5. Add token-baseline ablations that remove turn numbers, pressure/signal values, and history to
   identify whether its advantage is semantic or a shortcut.
6. Only after stable binary calibration, learn exact ambiguous counts 2 through 5 hierarchically.
7. Only after both gates pass, run oracle-count transition generation.

For a final generalization claim, generate a new untouched holdout after the method is frozen.
