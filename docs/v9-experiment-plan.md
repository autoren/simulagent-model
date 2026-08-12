# V9 neuro-symbolic evidence-grounding preregistration

## Objective

V9 tests whether a frozen language-model representation can ground natural
language evidence into determinant-specific state constraints. It does not ask
the model to learn transition algebra. The locked deterministic evaluator takes
the predicted allowed value sets and computes possible transitions and
identifiability exactly.

The V9 symbolic audit is a prerequisite: it reproduced all 6,480 exposed V8
simulator-oracle records with zero label, transition-count, or
compatible-assignment mismatches.

## Development corpus

Use only the six exposed V8 mechanics. For each selected semantic context,
render four natural-language template families and three lexical surfaces:

- inspection report;
- operator log;
- questioned claim with semantic negation;
- technical summary;
- canonical, entity-renamed, and paraphrased terminology.

Each observation contains seven independently recorded evidence units. Every
transition determinant has exactly one supporting unit; the remaining units are
noncausal distractors. Confirmed states are expressed through mechanic-specific
prose rather than the literal labels `active` and `inactive`. Unresolved states
are rendered as unknown-current, stale-only, or conflicting-current evidence.
The target for every determinant contains:

- allowed values: inactive, active, or both;
- temporal status: current, unknown-current, stale-only, or
  conflicting-current; and
- the exact supporting evidence-unit span.

The transition table is retained for deterministic scoring but withheld from
the grounding model input. It cannot help with the linguistic extraction task.

All surface and intervention variants of a semantic context stay in one
train/calibration split. A balanced noncausal scene identifier prevents two
observationally identical hidden assignments from creating cross-split prompt
duplicates; shortcut auditing must show that this identifier carries no useful
label signal.

## Holdout axes

Evaluation uses 13 fixed development folds:

1. one context-disjoint fold: train contexts versus calibration contexts;
2. six leave-one-mechanic-out folds;
3. four leave-one-template-family-out folds, evaluated on calibration contexts;
4. two leave-one-operator-family-out folds.

The operator families are fixed before generation:

- binary transition partitions: Hatch Traversal, Beacon Calibration, Pressure
  Hatch Relock;
- multiway transition partitions: Generator Tuning, Mirror Power Trip, Mirror
  Rejection.

Mechanic and operator holdouts may evaluate every record in the held-out group
because no record from that mechanic/operator family is used for training.
Template holdouts also require unseen calibration contexts.

## Pre-model gates

Before Qwen features are extracted, all structural validation must pass:

- zero context overlap, exact prompt overlap, conflicting duplicate, malformed
  span, or symbolic target mismatch;
- zero determinant identifiers or literal target labels in observation prose;
- metadata-only determinant-to-evidence match balanced accuracy no greater than
  0.60 in every fold;
- evidence-position-only match balanced accuracy no greater than 0.60 in every
  fold; and
- scene-code-only identifiability balanced accuracy no greater than 0.55 in
  every fold.

Character n-gram grounding, allowed-value, and temporal baselines are reported
as legitimate linguistic baselines and are not shortcut gates.

## Frozen grounding method

Use `mlx-community/Qwen3.5-0.8B-4bit`, frozen, with no adapter. For each candidate
determinant and each of the seven evidence units, form a grounding prompt that
contains the action, the complete determinant list, the queried determinant,
and that evidence unit. Extract the float32 mean of layer 6, with a maximum
sequence length of 512 and no permitted truncation.

Within each fold, fit three fixed linear logistic heads (`C=1.0`, seed 0,
class-balanced, LBFGS):

1. determinant/evidence match versus non-match;
2. allowed value set, trained only on gold matched pairs; and
3. temporal status, trained only on gold matched pairs.

For each determinant at evaluation, select the evidence unit with the largest
match score, then apply the value and temporal heads to that selected pair. No
gold span is supplied downstream. Feed the resulting allowed value sets into
the deterministic evaluator to obtain standalone identifiability.

## Metrics and hard gates

Report per fold and surface:

- evidence-unit span accuracy and token F1;
- determinant identification accuracy (equal to exact unit selection here);
- allowed-value-set exact accuracy;
- temporal-status accuracy;
- complete-ledger exact accuracy;
- symbolic identifiability balanced accuracy, F1, and AUC;
- possible-transition-set exact accuracy; and
- complete accuracy on label-flipping intervention pairs.

The fixed advancement gates are:

- context-fold span accuracy at least 0.70;
- every mechanic-fold span accuracy at least 0.60;
- every template-fold span accuracy at least 0.60;
- every operator-fold span accuracy at least 0.60;
- every fold allowed-values accuracy at least 0.65;
- every fold temporal accuracy at least 0.65;
- every fold downstream symbolic balanced accuracy at least 0.65; and
- every fold complete label-flip pair accuracy at least 0.60.

Every gate must pass. Mean performance cannot compensate for a failed fold.

## Firewall and decision rule

V9 reads no Tone Drift, V3 test record, prior holdout, V7 output, untouched V8
mechanic, or final V9 mechanic. It permits one development corpus generation,
one shortcut audit, one frozen feature extraction, and one complete 13-fold
linear evaluation. It permits zero LoRA runs and zero final-mechanic evaluations.

If every pre-model gate passes, the one frozen extraction/evaluation is
authorized. If every model gate then passes, V9 becomes eligible for a separate
final-mechanic protocol. Any failed model gate stops the current V9 method before
LoRA or final data.
