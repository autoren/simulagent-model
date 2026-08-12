# V8 ledger-derived decision diagnostic preregistration

## Motivation

The locked V8 structured-head run passed four of five gates. Across all 18
held-out mechanic-by-surface cells, mean balanced accuracy was 0.8796, paired
direction accuracy was 1.0, minimum determinant-status macro F1 was 0.70, and
decisive-determinant accuracy was 1.0. The only failure was minimum absolute
balanced accuracy: one cell scored 0.40 after applying a scalar threshold
calibrated on other mechanics.

This diagnostic tests a single task-derived correction to that calibration
failure. It does not retrain any model, choose a threshold, or alter a saved
head. The epistemic target is ambiguous exactly when at least one evidence row
is `UNRESOLVED_OUTCOME_SENSITIVE`; therefore the record decision should be
derived from the row ledger rather than from the auxiliary pointwise logit.

## Locked method

For every saved leave-one-mechanic-out Stage 4 head:

1. reproduce its five-way status logits for each evidence row;
2. compute each row's margin as its sensitive-status logit minus its largest
   non-sensitive-status logit;
3. use the maximum row margin as the record score; and
4. predict ambiguous iff that score is strictly greater than zero.

This is equivalent to predicting ambiguous when some row has the sensitive
status as its unique top class. The threshold is fixed by the multiclass
decision boundary and is not fitted on any records. Ties are non-ambiguous.

## Evaluation and gates

Evaluate the six saved heads on their respective held-out development mechanics
and all three locked surfaces. Retain the same five hard Stage 4 gates without
weakening them:

- minimum cell balanced accuracy at least 0.65;
- mean cell balanced accuracy at least 0.75;
- minimum cell paired score-direction accuracy at least 0.85;
- minimum cell row-status macro F1 at least 0.65; and
- minimum cell decisive-determinant accuracy at least 0.75.

No calibration split is read for threshold selection. Existing structured
metrics are recomputed only to verify exact reproduction of the saved heads.

## Decision and firewall

If every gate passes, the frozen-backbone structured classifier is eligible for
a separately locked evaluation on one newly constructed simulator-derived
final mechanic. If any gate fails, stop development before constructing or
evaluating that mechanic.

This lock permits one ledger-derived development evaluation, zero training
runs, zero adapter runs, and zero final-mechanic evaluations. Tone Drift, V3
test records, prior holdouts, V7 model results, and any new/untouched V8
mechanic remain prohibited.
