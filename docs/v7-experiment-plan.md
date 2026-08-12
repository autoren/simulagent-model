# V7 causal-evidence curriculum and untouched-mechanic plan

## Question and decision

V6 showed that the frozen 0.8B representation transferred weakly because evidence wording was
conditionally predictive of the label in development and reversed on mirror rejection. V7 asks
whether removing that shortcut before model access yields stable transfer to a genuinely new
causal mechanic. LoRA remains ineligible unless every frozen development and untouched-mechanic
gate passes.

## Data firewall

- V7 generation reads the simulator source and fresh scenario seeds only. It reads no V3 record,
  V5 challenge record, V6 mirror-rejection holdout record, prior holdout score, or prior model
  result.
- The final development revision uses fresh seeds 9601–9612 for short-start relock and power trip.
  The untouched tone-drift mechanic uses disjoint seeds 9801–9804. An earlier development-only
  attempt failed its frozen calibration gate and never opened or scored its reserved mechanic.
- Installing the mirror can detune the carried tuning fork without changing mirror seating or
  generator power. The recovery action is to retune it at the generator coil bank. This item-state
  reversal is implemented in the simulator and is absent from every prior corpus.
- The untouched mechanic may be generated and hash-locked, but its records are not opened by the
  shortcut audit, feature extractor, trainer, or probe freezer. It is scored exactly once.

## Curriculum contract and pre-model rejection

Every base input has canonical, entity-renamed, and paraphrased views with one shared binary target.
The target contains only `ambiguous` and the surface-invariance relation. Possible outcomes, counts,
oracle traces, empirical support, privileged state, and transition rules are excluded.

Development includes simulator-derived label-changing evidence pairs plus relational status
ledgers. Each ledger contains hatch, generator, mirror, and weather roles with exactly two
`confirmed` and two `unresolved` values. Its token multiset is identical across labels; only the
role-to-status assignment changes. The same assignment has opposite labels for relock and power
trip because different state roles control their candidate actions. This makes the intended signal
causal and relational rather than lexical. Selection balances every mechanic × evidence rung ×
action template × surface cell. Connected contexts and evidence groups are assigned before
sampling, so no context, prompt, or evidence group crosses train and calibration.

Before Qwen features may be extracted:

1. The maximum conditional label-rate gap must be at most 5 points.
2. A metadata-only lookup over mechanic, evidence rung, action template, and surface must score at
   most 55% calibration balanced accuracy.
3. An evidence-card-only token Naive Bayes model must score at most 60% calibration balanced
   accuracy and at most 0.65 AUC.

Any failure rejects the corpus before model access.

## Frozen baseline

The method is fixed to Qwen3.5-0.8B-4bit, full input, layer 6 mean pooling, and a class-balanced
float32 SAGA logistic head with `C=10` and seed 0. Complete triplets train with equal total weight.
The threshold is fitted once on canonical calibration. No model size, layer, pooling rule,
regularization, input variant, threshold, gate, or stratum is selected using tone drift.

The dataset, shortcut report, implementation, gates, development features, probe coefficients,
scaler, threshold, and calibration result must be hash-locked before the untouched records are
opened by the evaluator.

## Preregistered LoRA gates

All gates must pass:

1. Canonical development calibration balanced accuracy is at least 75%.
2. Untouched canonical balanced accuracy is at least 70%.
3. Its context-group bootstrap lower bound is at least 55%.
4. Entity-renamed and paraphrased balanced accuracy are each at least 65%.
5. Canonical prediction agreement with each transformed surface is at least 85%.
6. Complete surface-triplet accuracy is at least 60%.
7. Both members of an oracle label-changing pair are classified in the correct direction at least
   75% of the time.
8. Ambiguous pair members receive higher scores than identifiable members in at least 75% of
   comparisons.
9. Worst supported evidence, action, or evidence × action stratum balanced accuracy is at least
   55%.

Failure of any gate is a no-go for LoRA. Passing all gates permits only the next controlled LoRA
plus float32 classification-head experiment; it does not validate exact counts, transition
generation, closed-loop recovery, or behavioral belief revision.
