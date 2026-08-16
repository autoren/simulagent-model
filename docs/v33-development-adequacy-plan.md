# V33 development protocol: learning adequacy before another sealed suite

## Purpose

V32 did not adequately fit its own training population. V33 is therefore an exploratory but
bounded development study. It asks whether the frozen V32 representation can support the required
atom, lexical-sign, and outer-operation variables under competent head optimization, and whether
joint training causes negative interference.

V33 makes no new held-out generalization claim. It may use only V32 `factor_fit` and
`factor_calibration` records and their already-extracted frozen features. Both V32 evaluation
strata, every V32 evaluation prediction, V28, and all transition/program signals are forbidden.

## Bounded learning-curve search

One development seed trains five objectives: atom, direct truth, lexical sign, outer operation,
and the six-loss joint-auxiliary parser. Each objective is trained at exactly three learning rates.
Each path runs for 16 epochs and is reported at epochs 1, 2, 4, 8, and 16. This is 15 training
paths and 75 checkpoint reports; no path is added after results are observed.

The selected checkpoint for each objective maximizes its registered calibration metric, then its
fit counterpart, then a registered component mean. Exact ties prefer fewer epochs and then the
lower learning rate. Calibration is an unseen-surface development set, not a sealed test.

## Confirmation and interference diagnosis

Only the five deterministically selected configurations are retrained from scratch under three
fresh confirmation seeds. Their outputs form four systems:

- independent-direct: atom and direct truth from independently optimized modules;
- independent-compiled: the same atom module plus independent sign and operation modules and the
  unchanged deterministic compiler;
- joint-direct: all outputs from one joint-auxiliary head, using its direct truth output;
- joint-compiled: the identical joint head, using its sign and operation outputs with the fixed
  compiler.

Thus direct versus compiled comparisons share all trained artifacts within each architecture.
Independent versus joint comparisons locate multitask interference without changing the frozen
representation.

## Qualification

Qualification is deliberately stricter than ordinary exploratory success. Every one of the three
confirmation seeds must pass all registered fit and calibration component gates. A qualified
joint parser is preferred for simplicity unless the independently optimized parser exceeds its
mean calibration compiled exact-fact accuracy by at least 0.05.

Failure branches are fixed in advance:

- fit failure means the current representation/head optimization remains inadequate;
- fit success with calibration failure means surface-family transfer remains inadequate;
- independent success with joint failure identifies negative multitask interference;
- joint success qualifies the shared factorized interface for a future fresh sealed study.

Even qualification does not authorize reuse of V32 evaluation or V28. It authorizes only a new
preregistration and generation of entirely fresh language families and constructions.

## Integrity limits

V33 performs no backbone forward pass, adapter training, V28 replay, or fresh-suite construction
before qualification. The source feature artifact, allowed corpus files, implementation, search
space, selection rule, gates, and result audit are hash-locked before training begins.
