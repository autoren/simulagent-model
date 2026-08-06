# V4 binary identifiability experiment plan

## Question

Does reducing V3's five-way outcome-count target to a balanced binary target, then selecting a
checkpoint and logit threshold on a separate context-disjoint calibration fold, produce stable
identifiability discrimination across random seeds?

## Frozen data firewall

- Source: dataset V3 (`fe62d8bb3877792d301bb6907abc2bb52df7cd2ccce3f6ffcfc9c70b7f1ed6e3`).
- Training and calibration are formed only from V3 training context groups.
- V3 validation is reserved for one evaluation of each frozen seed/checkpoint/threshold tuple.
- V3 test is not read, copied, scored, or used for any decision.
- A new untouched holdout remains necessary for any final generalization claim.

The deterministic V4 split contains 1,037 training records in 131 groups, 181 calibration
records in 23 groups, and 154 validation records in 19 groups. The maximum ambiguity-rate gap is
1.13 percentage points, the maximum mechanic-share gap is 3.78 points, and no prompt or context
crosses a split.

## Minimal controlled change

The model and optimization settings remain those of V3: Qwen3.5-0.8B-4bit, 16 LoRA layers,
3.608M trainable parameters, batch size 1, 400 updates, `1e-5` learning rate, 1,024-token cap,
and checkpoints every 100 updates. Training records are oversampled to equal identifiable and
ambiguous class counts.

The output changes from five digit candidates to two single-token labels:

- `A`: exactly one supported transition (identifiable).
- `B`: multiple supported transitions (ambiguous).

For each checkpoint, the ambiguity score is `logit(B) - logit(A)`. Exactly one threshold is
fitted on calibration. The checkpoint is selected by calibration balanced accuracy, ambiguity
F1, ROC AUC, and then earlier step. Only the selected checkpoint and frozen threshold are applied
to validation.

## Seeds and gates

Seeds are fixed at 0, 1, and 2.

Engineering stability passes only if:

1. Every seed exceeds 50% validation balanced accuracy.
2. At least two seeds reach 60%.
3. Mean validation balanced accuracy reaches 60%.
4. The seed range is at most ten percentage points.
5. Every seed predicts both classes on validation.

Scientific usefulness is reported separately. It requires every seed and their mean to match the
primary full-input token Naive Bayes baseline trained on the same V4 training fold with its
threshold fitted on the same calibration fold. Ablations are diagnostic and are not eligible to
replace that preselected primary reference.

## Token-baseline ablations

The fixed diagnostic variants are full input, remove turn, remove pressure and signal, remove
history and memories, remove all three state scalars, and remove both scalars and history. Each
variant is trained only on V4 training and thresholded only on calibration. These variants test
whether token performance depends on numeric state values or observation history.

## Decision after V4

- If engineering stability fails, stop generative-label LoRA work and test a dedicated linear or
  multilayer classification head on frozen and LoRA-adapted hidden representations.
- If engineering stability passes but scientific usefulness fails, treat binary SFT as functional
  but inferior to a simple token classifier; prioritize representation probes and corpus variation.
- If both pass, proceed to exact counts 2–5 among ambiguous examples before transition generation.
