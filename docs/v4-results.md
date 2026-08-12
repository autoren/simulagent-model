# V4 binary identifiability results and decision

## Decision

The V4 binary-label revision failed both its preregistered engineering stability gate and its
scientific token-baseline gate. Exact ambiguous-count training and oracle-count transition
generation remain gated off.

V4 did resolve one V3 failure mode: after calibration, every selected checkpoint predicted both
classes rather than remaining a constant digit predictor. It did not, however, learn a stable or
useful ordering of examples. Mean validation balanced accuracy was 51.20%, and only seed 2 rose
modestly above chance at 55.49%.

The next experiment should not be another generative vocabulary-label run or a larger model run.
It should test a dedicated float32 classification head over frozen hidden representations first,
then add LoRA only if the frozen probe establishes useful representational signal.

## Data and measurement firewall

V4 deterministically subdivided only V3 training contexts into training and calibration. V3
validation was evaluated once per frozen seed/checkpoint/threshold tuple, and V3 test was never
read.

| Check | Result |
| --- | ---: |
| Train / calibration / validation records | 1,037 / 181 / 154 |
| Train / calibration / validation context groups | 131 / 23 / 19 |
| Ambiguity rates | 40.02% / 39.78% / 40.91% |
| Maximum ambiguity-rate gap | 1.13 points |
| Maximum mechanic-share gap | 3.78 points |
| Prompt overlaps | 0 |
| Context overlaps | 0 |
| V3 test records read | 0 |
| V4 dataset SHA-256 | `3c5d51d6d7620e589d7ddbfe6ce5667ac9bd21e601abe31436e27eefa9e69d88` |

## Controlled model change

The base model, LoRA placement, optimizer, learning rate, sequence cap, checkpoint cadence, and
three seeds were unchanged from V3. Training was balanced and the target was reduced to two
single-token labels: `A` for identifiable and `B` for ambiguous. Each checkpoint's score was
`logit(B) - logit(A)`. Checkpoint and threshold selection used calibration only.

| Seed | Selected step | Calibration balanced | Validation balanced | Validation F1 | Validation AUC | Predictions (I/A) |
| ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 0 | 200 | 51.66% | 49.15% | 47.74% | 0.491 | 62 / 92 |
| 1 | 300 | 50.41% | 48.96% | 54.17% | 0.492 | 25 / 129 |
| 2 | 200 | 54.21% | 55.49% | 51.06% | 0.555 | 76 / 78 |

Mean validation balanced accuracy was 51.20%, with a 6.53-point seed range. Both gates failed:

- Engineering: two seeds were below 50%, no seed reached 60%, and the mean was below 60%.
- Scientific: the mean and every-seed requirements failed against the 58.24% primary full-input
  token baseline.

All selected checkpoints were nonconstant and no prompt was truncated. Consequently, V4 rules
out truncation, class-frequency shift, a fixed threshold, and five-way label fragmentation as
sufficient explanations for V3's failure.

## Loss again concealed the failure

All seeds reduced full-calibration teacher-forced loss from 3.647 initially to roughly 0.11–0.13,
while selected validation AUC remained 0.49–0.55. Seed 1 achieved lower loss than seed 0 but
slightly worse validation ranking. As in V3, label-token loss measured successful optimization of
the output interface, not stable example-level discrimination.

## Coarse score resolution

The selected seed margins contained only two or three distinct values on both calibration and
validation:

| Seed | Distinct validation margins | Range |
| ---: | ---: | --- |
| 0 | 2 | 0.250 to 0.375 |
| 1 | 3 | -0.125 to 0.125 |
| 2 | 2 | -0.500 to -0.375 |

Candidate logits around magnitude 22 were emitted on coarse increments before conversion to
Python floats. Threshold fitting could divide these bins, but it could not recover a fine-grained
ranking that the vocabulary output head had not represented. This means V4 does **not** establish
that Qwen's hidden states lack epistemic signal. It establishes that the current low-precision
language-model head plus binary SFT did not expose useful signal.

See `docs/v4-score-diagnostics.md` for the reproducible score-bin report.

### Post-hoc float32 projection audit

A no-retraining audit recomputed each selected checkpoint's A/B margin by projecting the final
hidden state through the two vocabulary-head rows in float32. This recovered 153–154 distinct
validation scores per seed instead of two or three, confirming that bfloat16 vocabulary logits
were a real resolution bottleneck. It did not rescue discrimination: mean validation balanced
accuracy was 51.16% and mean AUC was 0.540. Coarse output precision therefore contributed to V4's
failure but was not its main cause.

## Token-baseline ablations

The primary full-input token Naive Bayes model reached 58.24% validation balanced accuracy with a
threshold fitted on calibration and a validation AUC of 0.714. Removing history and memories
reached 76.92% balanced accuracy, while removing only turn, pressure, or signal did not improve
the fitted full-input result.

The no-history result is a diagnostic, not a replacement for the preregistered primary baseline.
It may reflect noisy repeated history, input-length effects in raw Naive Bayes scores, or a
dataset shortcut. Calibration AUC remained near chance for every ablation even when validation
AUC was much higher, showing that label and mechanic stratification alone did not align all token
correlations. Entity renaming, paraphrase pairs, evidence-rung holdouts, and complete-mechanic
holdouts are still required before interpreting token performance as semantic epistemic reasoning.

## What V4 establishes

1. Binary targets and a separate threshold do not rescue the current generative LoRA approach.
2. The adapter can move examples into more than one output bin, but its ranking is weak and
   unstable across seeds and checkpoints.
3. Teacher-forced loss is again unsuitable for model selection on this task.
4. The selected vocabulary-logit margins are too coarse to function as a high-resolution
   classifier.
5. Visible-input token signal remains materially stronger than the selected LoRA results.

V4 does not establish that a larger model would fail, that hidden representations contain no
signal, or that the intended behavioral belief-revision problem is unlearnable. The current task
is still static epistemic classification, not closed-loop recovery or perseveration.

## Next experiment

The next controlled sequence should be a discriminative representation probe. The 0.8B probe is
run first, with larger models conditional on whether the smallest model exposes useful signal:

1. Extract pooled hidden representations from frozen Qwen3.5-0.8B, 4B, and 9B models.
2. Train a class-balanced float32 linear head with regularization selected on calibration.
3. Evaluate the frozen configuration once on the development validation partition across seeds.
4. Compare full input and the fixed no-history diagnostic without selecting between them on
   validation.
5. If a frozen probe succeeds, train LoRA plus the same classification head. If it fails across
   model sizes and layers, revise serialization and corpus variation before further fine-tuning.
6. After the method is frozen, generate a new untouched holdout with entity renamings, paraphrases,
   held-out mechanics, and evidence-rung minimal pairs.

Only after stable binary discrimination should the project learn counts 2–5 hierarchically or
return to transition generation. Belief-update and recovery-policy datasets remain later,
separately measured stages tied back to the original behavioral simulator.
