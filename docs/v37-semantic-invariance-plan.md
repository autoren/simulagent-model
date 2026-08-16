# V37 preregistration: candidate-conditioned semantic invariance

## Rationale

V36 transferred predicate identity, entity binding, exact atoms, and relation order perfectly, but
lexical sign and outer-operation accuracy both fell to about 0.768 on new surface families. Most
operation errors defaulted to unresolved, and operation accuracy fell from 0.931 on clean clauses
to 0.509 with distractors. The fixed V35/V36 readouts therefore appear to encode local surface
correlations rather than stable semantic compatibility.

V37 asks whether the frozen 4B grounder contains more transferable semantic information when it
judges explicit candidate analyses one at a time. It is a development repair, not a new scientific
claim and not an end-to-end relational experiment.

## Frozen scope and firewall

The backbone, ontology, atom/binding path, truth compiler, and executor remain unchanged. V37 may
train only on V32 `factor_fit` and the now-exposed V36 confirmation corpus. It must not read V32
calibration or evaluation records, run V28, train an adapter, change the backbone, or construct an
end-to-end relational suite.

All prompts, candidates, validation templates, generator rules, sampling quotas, methods, alphas,
metrics, gates, and decision rules are hash-frozen before feature extraction or fitting.

## Candidate-conditioned interface

For every clause the model receives two lexical-sign hypotheses and five outer-operation
hypotheses. Each prompt asks whether one fully defined candidate analysis is compatible with the
evidence. No prompt contains a target label. The final hidden state and native Yes-minus-No logit
margin are retained for each candidate.

Four preregistered methods are compared:

1. a shared balanced binary ridge over candidate-conditioned hidden states;
2. a balanced multiclass ridge over each record's vector of candidate Yes/No margins;
3. zero-fit argmax over native Yes/No margins; and
4. the prior direct hidden-state ridge, refit on the same development population.

Sign and operation methods are selected separately using only five-fold grouped cross-validation
on the 400-record fit population. Grouping keeps each source/operation/surface family within one
fold. Selection maximizes mean record accuracy, then worst-fold accuracy, then stronger
regularization. Validation selects nothing.

## Fresh development validation

The validation set contains ten new surface families: two for each of the five operations, with
both lexical signs. Its 360 clauses cover direct and inverse relations, reversed arguments, clean
evidence, prefix distractors, and a novel suffix-distractor placement. No exact evidence text or
normalized surface template may overlap V32 or V36.

The frozen model performs exactly nine forwards per fit and validation record: seven candidate
views and two direct-baseline views. Across 760 records the fixed budget is 6,840 forwards. The
validation set is scored once after fit-only method and alpha selection.

## Gates and next decision

Qualification requires validation sign, operation, and compiled-truth accuracy of 0.95;
worst-operation accuracy of 0.90; worst-family truth of 0.85; distractor and negative-composition
truth of 0.90; every registered pair category of 0.85; and at least 0.15 compiled-truth gain over
the untouched frozen V36 interface.

- If every gate passes, preregister—but do not construct—a fresh semantic-only confirmation.
- If candidate conditioning materially improves transfer but misses a gate, continue development
  on the isolated semantic interface.
- If it provides no material gain, stop linear prompt/readout iteration and pivot to a constrained
  semantic parser or a separately justified stronger frozen grounder.

No V37 outcome authorizes V32 evaluation access, V28, adapter training, or an end-to-end suite.
