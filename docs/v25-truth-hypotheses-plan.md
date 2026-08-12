# V25 protocol: explicit truth-hypothesis compatibility

## Objective and status

V25 is an exposed-data development experiment that isolates V24's remaining truth-semantic
failure. V24's candidate assignments are immutable inputs: no atom-matching feature, head, score,
proposal, or assignment may be refitted or revised. V25 creates no fresh benchmark records and
cannot support a holdout or final claim.

The motivating zero-fit diagnostic found that oracle gold-pair truth accuracy remains 0.876, nearly
identical to the end-to-end 0.878. The dominant failure is the held-out `eval_c` contrastive true
cell: the evidence states the false alternative first and the true fact second, but V24's direct
multiclass candidate-span readout predicts false in 156 of 157 cases.

## Truth-only representation

For every evidence unit, the fixed V24 assignment chooses the candidate used at inference. Clean
truth supervision additionally includes the gold candidate in `grounding_fit`; it is the same row
when V24 assigned correctly. Each evidence/candidate pair is rendered three times with one final
assessment hypothesis:

- the evidence entails the candidate fact;
- the evidence contradicts the candidate fact;
- the evidence leaves the candidate fact unresolved.

The layer-8 tokens of the final assessment hypothesis are mean-pooled in float32. Evidence and the
candidate fact occur before the hypothesis, allowing the frozen causal representation to bind
negation and contrastive clause order directly to an explicit semantic relation.

## Fixed head and inference

A single balanced C=1 `liblinear` head with random state 2501 is fitted once on all three hypotheses
for each gold candidate in `grounding_fit`. The correct assessment is positive and the other two
are negative. At inference, the truth status is the hypothesis with maximum compatibility
probability. Calibration is report-only; it selects nothing.

The V24 one-to-one candidate assignment is copied byte-for-byte, paired with the V25 truth status,
and evaluated through the unchanged grounding metrics, schema inducer, DSL, and executor.

## Decision

Passing every development gate authorizes freezing the combined V24-match/V25-truth interface
before constructing a genuinely fresh relational surface benchmark. A truth-gate failure rejects
this factorization. A truth pass with downstream failure localizes the remaining issue to exact
graph assembly or symbolic integration. No result authorizes LoRA, grammar expansion, or reuse of
the exposed V22r2 corpus as a final evaluation.
