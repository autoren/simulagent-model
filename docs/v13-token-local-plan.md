# V13 preregistration: 4B token-local relation diagnostic

## Decision being tested

V12 showed that neither a signed linear comparison nor a fixed joint MLP can orient active/inactive relations consistently from the generic final token, even at 9B. V13 performs the single token/span interaction extraction authorized by that result.

The experiment uses only the pinned V11 4B backbone at the same homologous layer 8/32. This is the smallest model that passed every V11 evidence-span gate, and V11/V12 found no monotonic polarity benefit at 9B. There is no scale, layer, prompt, or hyperparameter sweep.

## Locked extraction

V13 reuses the exact 6,984 V10 NLI prompt strings and their locked order. It extracts two float32 readouts from the bfloat16 layer-8 hidden states:

- `hypothesis_last`: the final tokenizer token whose character offsets overlap the current-state hypothesis; and
- `hypothesis_mean`: the mean over all tokenizer tokens overlapping that hypothesis.

The evidence precedes the hypothesis in the causal prompt. Therefore the last hypothesis token has attended to the evidence and the complete hypothesis, unlike an evidence-span token, while avoiding the generic assistant-generation suffix used by V10–V12. Offset-derived spans, prompt hashes, token limits, and zero truncation are mandatory.

## Locked evaluation

The 7,380 eligible examples, labels, and 24 folds are identical to V12: only gold-matched evidence with gold `CURRENT` status, exactly balanced between inactive and active. The two preregistered linear comparisons run once each:

1. `hypothesis_last_linear`: active last-token vector minus inactive last-token vector;
2. `hypothesis_mean_linear`: active hypothesis-mean vector minus inactive hypothesis-mean vector.

Both are balanced logistic regressions with `C=1`, `lbfgs`, 3,000 maximum iterations, and seed zero plus fold index.

If and only if neither linear representation passes, one 32-unit ReLU MLP runs over `concat(last_difference, mean_difference, last_pair_mean)`. Its fixed optimizer settings match V12: Adam, alpha `0.001`, batch size `256`, learning rate `0.001`, 200 maximum iterations, tolerance `1e-4`, no early stopping, and 20 iterations without improvement. There is no tuning from V13 results.

The pass rule remains accuracy at least 0.70 in every fold and 0.65 in every non-empty fold-by-surface cell. Balanced accuracy, ROC AUC, and swap-complement accuracy are diagnostic.

## Decisions

- If `hypothesis_last_linear` passes, use it as the fixed 4B polarity comparator in a separately locked temporal-repair experiment.
- Otherwise, if `hypothesis_mean_linear` passes, use that fixed comparator.
- Otherwise, if the conditional MLP passes, freeze that nonlinear comparator and repair temporal transfer separately.
- If all three fail, stop frozen feature probing. The next research plan must change supervision or the language-grounding objective; it must not add another scale, layer, token pooling, or post-hoc classifier. LoRA is not automatically authorized by failure.

No V13 outcome authorizes final-mechanic access. A successful polarity comparator would still require a repaired temporal head and another complete symbolic pipeline evaluation.

## Firewall

V13 permits one 4B extraction over the existing NLI prompts, two linear 24-fold evaluations, and at most one conditional MLP 24-fold evaluation. It permits zero adapter runs, hyperparameter searches, alternative layers, alternate prompts, final-mechanic evaluations, or protected data access.
