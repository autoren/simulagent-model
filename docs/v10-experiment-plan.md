# V10 current-state polarity preregistration

## Objective and scope

V10 tests the linguistic component left unresolved by V9. The frozen model must
first match evidence to a transition determinant, then classify temporal status,
and only for reliable current evidence determine whether the text entails the
active or inactive state. All non-current statuses deterministically permit both
Boolean values. The locked V9 symbolic evaluator remains the sole authority for
transition enumeration and identifiability.

V10 is a development experiment over the same six exposed V8 mechanics. It
reads no final mechanic, Tone Drift, V3 test set, prior holdout, or V7 model
output. It permits no adapter training and no larger-model extraction.

## Corpus and targets

Use replica zero from every V8 semantic context and render each intervention
member and lexical surface under six language families:

1. direct assertion;
2. explicit negation of the opposite state;
3. denial of the opposite claim;
4. rejection of the opposite report;
5. contrastive correction; and
6. a scoped relative-clause rejection.

Thus every leave-one-template fold has four other negation constructions in
training. Each determinant supplies two counterfactual hypotheses in fixed,
unlabelled order: the mechanic-specific active proposition and inactive
proposition. Current evidence has complementary `ENTAILED`/`CONTRADICTED`
targets; unknown-current, stale-only, and conflicting-current evidence has two
`UNKNOWN` relations. The two hypotheses form a minimal contradiction pair.

Allowed values are never independently labelled by a learned head. They are
derived by the following fixed rule:

- any non-current temporal status -> `{inactive, active}`;
- current + entails-active/contradicts-inactive -> `{active}`;
- current + contradicts-active/entails-inactive -> `{inactive}`;
- every other current relation pattern -> `{inactive, active}`.

Semantic contexts are split by complementary Boolean assignments. Complement
pairs stay together, which makes current active/inactive targets exactly
balanced in both train and evaluation wherever an evaluation complement exists.
One-determinant Hatch Traversal has only one complement group and is therefore
excluded from the context-disjoint evaluation split; it remains covered by its
strict leave-one-mechanic-out fold.

## Frozen representations

Use frozen `mlx-community/Qwen3.5-0.8B-4bit` layer 6, float32 pooling, no
truncation, logistic heads with `C=1.0`, class balancing, and seed zero.

The a-priori primary is `nli_final`:

- evidence matching and temporal status use evidence-span-pooled pair prompts;
- each active/inactive hypothesis gets a separate NLI-style prompt; and
- its final layer-6 token feeds a three-way relation head.

Two diagnostics are evaluated on the same folds but cannot replace the primary
after results are observed:

- `mean_direct`: full-prompt mean pooling and a direct current-polarity head;
- `evidence_span_direct`: evidence-span pooling and the same direct head.

## Holdouts and oracle ablations

The final evaluation contains 24 fixed folds:

- one context-disjoint fold;
- six leave-one-mechanic-out folds;
- six leave-one-template-family-out folds;
- three leave-one-state-lexicon-out folds;
- two operator-family folds restricted to generic enabled/disabled state
  language, so operator transfer is not confounded with unseen state wording;
- six combined operator-family plus state-lexicon folds.

Every final fold trains only on non-evaluation contexts unless the held mechanic
is entirely absent from training. Combined folds exclude both the held operator
and held lexicon from training. Results are reported overall and for every
non-empty lexical surface cell.

For every representation and fold, report four causal ablations:

1. oracle span + oracle temporal;
2. predicted span + oracle temporal;
3. oracle span + predicted temporal; and
4. fully predicted.

These distinguish polarity failure from span-selection and temporal-error
cascades. Also report hypothesis-pair consistency, exact ledgers, possible
transition sets, symbolic balanced accuracy, per-surface intervention pairs,
and complete six-record intervention groups.

## Pre-model gates

Before Qwen access, require zero validation, span, split-overlap, prompt-
duplicate, target-derivation, or symbolic mismatch; exact current-state balance
within every mechanic/template/surface cell; and no determinant identifiers or
literal target labels in observation prose.

In every defined fold, require:

- metadata-only match balanced accuracy <= 0.60;
- position-only match balanced accuracy <= 0.60;
- metadata-only current-polarity balanced accuracy <= 0.55; and
- hypothesis-position-only relation balanced accuracy <= 0.55.

Character n-gram linguistic baselines are report-only.

## Primary hard gates and decision rule

Only `nli_final` controls advancement. Every aggregate fold must reach span
accuracy 0.65, temporal accuracy 0.70, oracle-span/oracle-temporal polarity
accuracy 0.70, NLI pair consistency 0.70, fully predicted allowed-value accuracy
0.65, symbolic balanced accuracy 0.65, complete flip-pair accuracy 0.60, and
complete six-record intervention-group accuracy 0.50. Every non-empty surface
cell must reach respectively 0.60, 0.65, 0.65, 0.65, 0.60, and 0.60 for the
first six applicable measures.

If all primary gates pass, stop at frozen 0.8B: neither scaling nor LoRA is
justified. If oracle-span/oracle-temporal polarity or NLI consistency fails,
authorize a separately locked frozen 4B/9B capacity diagnostic, not LoRA. If
oracle polarity passes but the full pipeline fails because span or temporal
status fails, revise only that upstream component. LoRA can be proposed only
after a larger frozen comparison demonstrates transferable polarity signal and
must remain restricted to linguistic grounding; it is not authorized by V10.
