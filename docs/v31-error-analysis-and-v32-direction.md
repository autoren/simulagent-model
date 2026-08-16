# V31 error decomposition and V32 direction

## Bottom line

V31 is a clean negative for both registered learned systems, but it is not evidence that the
frozen representation lacks relational grounding information. The frozen structured readout
recovers predicates, entities, and directed relation order well, then fails primarily when it must
compose a lexical fact with negation or denial scope. The registered LoRA recipe damages all of
those abilities and should not be tuned on the exposed V31 evaluation suite.

The next fresh experiment should therefore test an explicitly factorized frozen semantic parser,
not another adapter rank or learning-rate sweep and not another graph-search layer.

## What V31 established

The one-shot evaluation contained 1,300 clauses from 25 unseen surface families. Every metric was
reproduced from saved predictions, all 10,400 planned evaluation forward passes were accounted
for, and the post-result integrity audit passed.

| System | Predicate | Argument 1 | Relation order | Truth | Exact fact | Exact scene |
|---|---:|---:|---:|---:|---:|---:|
| Zero-shot reference | 0.985 | 0.928 | 0.376 | 0.567 | 0.341 | 0.090 |
| Frozen structured readout, mean | 0.967 | 0.980 | 0.961 | 0.814 | 0.783 | 0.617 |
| LoRA plus the same readout, mean | 0.222 | 0.508 | 0.254 | 0.493 | 0.103 | 0.033 |

Neither learned system passed. No language system was selected, and the conditional V28 replay
was therefore not authorized.

The LoRA-minus-frozen exact-fact difference was -0.680. Its surface-family bootstrap 95% interval
was [-0.764, -0.587], and every one of the 25 family differences was negative. This supports a
strong conclusion about the tested final-eight-layer, rank-8, learning-rate-0.0002 recipe: it causes
negative transfer. It does not prove that every possible adapter configuration would fail.

## Where the frozen system fails

Across all three frozen seeds, 3,052 of 3,900 facts were exactly correct. Of the 848 errors, 694
(81.8%) had the predicate, argument 1, and argument 2 all correct and only the truth status wrong.
Thus most residual errors occur after the system has already identified the relevant graph atom.

| Semantic operator | Mean exact fact | Mean truth accuracy |
|---|---:|---:|
| Affirmative | 0.991 | 1.000 |
| Contrastive | 0.924 | 0.938 |
| Explicit unknown | 0.909 | 0.999 |
| Double negation | 0.545 | 0.550 |
| Negated opposite | 0.544 | 0.581 |

This is not a general inability to distinguish `false` from `unknown`: explicit-unknown truth is
nearly perfect. It is also not mainly relation direction: relation-order accuracy is 0.961. The
dominant failure is transferring the correct polarity through unfamiliar denial and double-
negation constructions. The controlled affirmative/double-negation and affirmative/negated pairs
both average only 0.542 pair-exact accuracy.

## V32 scientific question

V32 should ask:

> Under the same declared ontology and supported operator inventory, does explicitly factorizing
> atom grounding from polarity-and-scope composition transfer to unseen surface families better
> than a monolithic frozen signed-fact readout?

This is narrower and better supported than asking whether more parameter adaptation can rescue the
whole system. It directly tests the error boundary exposed by V31.

## Proposed factorized interface

Keep the backbone frozen and keep typed predicate/entity grounding. Replace the direct
three-way truth head with supervised intermediate decisions whose composition is deterministic:

1. identify the canonical predicate and typed arguments;
2. identify whether the embedded literal is the positive or negative lexical form of that atom;
3. identify the outer evidence operation, such as assertion, rejection/denial, contrastive
   selection, double denial, or unresolved status; and
4. compile those decisions to `true`, `false`, or `unknown` with a fixed truth table.

The intermediate targets are generator-derived and may be used only on fit data. At evaluation,
the agent receives the same ontology, entities, and evidence text as before; it is not given the
operator or lexical-polarity labels.

## Required controls

V32 should use a newly generated family-disjoint corpus. V31 evaluation surfaces are now exposed
and may be used only for descriptive development, never as V32 confirmation data.

The primary comparison should be:

- a frozen monolithic structured readout equivalent to V31; and
- a frozen factorized parser with a matched representation, fit population, seed order,
  optimization budget, and approximately matched head capacity.

The suite should preserve direct/inverse, argument-reversal, distractor, false/unknown, and
cross-operator equivalence pairs, while adding scope-sensitive pairs that hold the embedded literal
fixed and change only the outer operation. Report intermediate lexical-polarity and scope accuracy
in addition to the existing signed-fact metrics.

Preregister one corpus, one feature extraction, fixed seeds, no checkpoint selection, and one
sealed evaluation. Do not include LoRA in the primary V32 experiment. Do not run V28 unless one
language system passes all unchanged end-to-end signed-fact gates.

## Decision rule

- If the factorized parser passes and materially exceeds the matched monolithic head, adopt the
  compositional language interface and run the one authorized V28 replay.
- If both pass, prefer the simpler monolithic head unless factorization clears a preregistered
  material-advantage rule.
- If both fail primarily at scope classification, the fixed-ontology language claim remains open;
  the next comparison should be an explicit grammar/constrained parser or a separately justified
  stronger frozen grounder, not an adapter sweep on the same suite.
- If atom grounding rather than scope becomes the dominant error, revisit representation locality
  and entity binding before touching program induction or graph marginalization.
