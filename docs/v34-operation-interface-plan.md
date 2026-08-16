# V34 preregistration: operation-focused frozen interface

## Scientific question

V33 showed that a strongly regularized linear readout can transfer predicate and lexical-sign
information across V32 surface families, while the five-way outer-operation decision remains the
dominant semantic failure. V34 asks whether this failure comes from the generic representation
prompt rather than from an absence of operation information in the frozen language model.

## Scope and firewall

This is an exposed-development diagnostic. It may read only `factor_fit` and
`factor_calibration`. It must not read either V32 evaluation split, any V32 evaluation features or
predictions, V28 signals, or a fresh suite. It performs no adapter or backbone training. A positive
result only authorizes further development of entity binding and modular assembly.

## Frozen comparison

The 4B frozen model receives the evidence text and five explicit, canonical operation definitions.
One forward pass per development clause yields the final hidden state and float32 logits for the
fixed labels A--E. Four methods are reported:

1. the legacy V32 final hidden state with a regularized ridge classifier;
2. the operation-focused final hidden state with the same classifier;
3. the five operation-prompt label logits with the same classifier; and
4. direct argmax over the native label logits, as a zero-fit reference.

The prompt never contains the target operation, lexical sign, predicate, arguments, candidate
fact, or final truth value.

## Selection

For each learned method and alpha, fit data are divided by the four surface names (`fit_a` through
`fit_d`). Selection uses leave-one-surface-name-out accuracy only: highest mean fold accuracy,
then highest worst-fold accuracy, then the largest alpha. The better of the two new prompt methods
is likewise fixed from fit cross-validation before calibration is scored. Calibration chooses no
prompt, method, alpha, threshold, or label mapping.

The registered budget is 75 ridge fits: 24 cross-validation fits plus one final fit for each of
three learned methods. The native argmax requires no fit.

## Decision

The selected new interface qualifies only if calibration operation accuracy is at least 0.90,
worst-operation accuracy is at least 0.80, oracle-lexical-sign compiled truth is at least 0.90,
and operation accuracy improves by at least 0.20 over the legacy hidden-state ridge baseline.

- If it qualifies, continue to regularized entity binding and modular development assembly.
- If it improves over the legacy baseline by at least 0.10 but misses a gate, continue operation-
  interface repair without opening a new suite.
- Otherwise, stop prompt/readout iteration and pivot to a constrained semantic parser or a
  separately justified stronger frozen grounder.

No V34 outcome authorizes V28, V32 evaluation reuse, or construction of a final suite.
