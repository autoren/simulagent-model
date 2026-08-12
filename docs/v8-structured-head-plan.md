# V8 structured action-conditioned head preregistration

## Authorization and purpose

The locked V8 Stage 3 diagnostic is the sole authorization for this stage. Its
leave-one-mechanic-out pair-difference accuracy was 1.0 for every mechanic and
surface, and its minimum pointwise pair direction was 0.75. The frozen
representation therefore contains a transferable ambiguity direction, while
absolute score calibration remains insufficient on several held-out surfaces.

Stage 4 tests whether an explicitly action-conditioned structured head can turn
that direction into an absolute ambiguity decision. It is a development-only
experiment. It does not train an adapter and it does not create, read, tune on,
or evaluate a final V8 mechanic.

## Locked data and folds

- Use only the six V8 development mechanics and the already frozen
  train/calibration artifacts.
- Run six leave-one-mechanic-out folds.
- Train a fold on the `train` split of the other five mechanics only.
- Select the ambiguity threshold on canonical `calibration` records from the
  other five mechanics only.
- Evaluate the held-out mechanic on canonical, entity-renamed, and paraphrased
  surfaces. Both of its stored splits are evaluation-only in that fold.
- Do not use Tone Drift, V3 test records, prior holdouts, V7 model outputs, or
  any new/untouched V8 mechanic.

## Frozen representations and structured inputs

The Qwen backbone is `mlx-community/Qwen3.5-0.8B-4bit`, frozen throughout.
Stage 3's locked float32 layer-6 mean representation is the record-level input.
A single additional locked extraction obtains layer-6 float32 mean embeddings
for the following deduplicated components:

1. candidate action,
2. complete action-dependency table,
3. each of the seven evidence-row roles, and
4. each of the seven evidence statements.

Component prompts contain only information already present in the model-facing
record. No target, simulator state, or answer-bearing annotation is added.
Components are capped at 512 tokens; any truncation is reported and is a
protocol failure for interpretation.

## Locked head

Project the record, action, dependency table, role rows, and evidence rows from
the frozen hidden width to 48 dimensions. Add the projected record, action,
table, role, and evidence representations per row, then apply GELU. A shared
five-way row head predicts:

- `RESOLVED_TRUE`,
- `RESOLVED_FALSE`,
- `UNRESOLVED_OUTCOME_SENSITIVE`,
- `UNRESOLVED_OUTCOME_INVARIANT`, and
- `IRRELEVANT`.

The ambiguity logit combines a learned pointwise head over the global/action/
table representations plus mean and max row summaries with 0.25 times the
log-sum-exp of the row-sensitive logits. No backbone parameter is trainable.

Train full-batch with Adam for exactly 300 steps, learning rate 0.003, seed 0
plus fold number, and no hyperparameter search. Use inverse-frequency weights
within each training fold. The locked loss is:

`1.0 * determinant CE + 0.5 * pointwise BCE + 2.0 * paired logistic + 0.1 * surface variance`.

Paired loss uses only oracle label-flip intervention pairs. Surface variance is
computed over the three surface renderings of each training scenario.

## Locked success gates

All gates are hard and are evaluated across all 18 held-out
mechanic-by-surface cells:

- minimum cell balanced accuracy at least 0.65;
- mean cell balanced accuracy at least 0.75;
- minimum cell paired direction accuracy at least 0.85;
- minimum cell row-status macro F1 at least 0.65; and
- minimum cell decisive-determinant accuracy at least 0.75.

The exact per-fold thresholds, losses, structured metrics, direction metrics,
and head weights are retained regardless of outcome.

## Decision rule

If every gate passes, V8 becomes eligible for a separately preregistered
evaluation on one newly created, simulator-derived final mechanic. Passing does
not authorize reading Tone Drift or any earlier final/test set, and it does not
authorize LoRA training.

If any gate fails, stop before constructing or evaluating a final mechanic.
Document the failing cells and revise the development method only under a new
lock.

## Run limits

- one structured-component extraction;
- one six-fold structured-head training/evaluation run;
- zero final-mechanic evaluations;
- zero adapter training runs.
