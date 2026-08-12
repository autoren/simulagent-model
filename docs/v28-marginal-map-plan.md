# V28 preregistration: marginal program MAP

V28 is one exposed-development, zero-forward comparison against V27. It tests whether V27's
remaining error is caused by Viterbi-style joint MAP choosing a spurious program supported by
one high-scoring graph, rather than aggregating evidence across all plausible support graphs.

## Frozen inputs

V28 reuses the locked V27 top-512 support graph construction, V24 match scores, V26/V27 native
truth scores, observed transitions, DSL, executor, and V26 query predictions. It performs no
model call, fit, threshold selection, score-weight selection, branch-budget selection, ontology
change, or fresh-benchmark access. The exploratory native-matching diagnostic is recorded as a
negative result and is not used as a score factor.

## Registered inference

For each support scene, softmax-normalize the retained V27 joint graph scores. For each program,
the trace likelihood is the summed probability of graphs under which the public observed
transition is possible. Multiply trace likelihoods under a uniform program prior. Select the
single maximum-posterior program, breaking ties by canonical program key. For each support
scene, select its highest-scoring graph compatible with that program. If every program has zero
likelihood, retain the V27 support graphs. Query predictions remain byte-identical to V26/V27.

This is marginal MAP, not a credible-set union: it preserves a single executable prediction and
therefore directly addresses V23's anti-widening failure.

## Interpretation

- Passing every registered gate authorizes a separately preregistered query-graph repair; it
  does not authorize a fresh benchmark or a broader world-model claim.
- Improvement over V27 without a full pass supports marginal program evidence but keeps the
  work in exposed development.
- No improvement localizes the residual to intrinsic support identifiability or score
  calibration. It does not authorize LoRA automatically.
