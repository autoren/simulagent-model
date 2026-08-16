# V30 protocol: sealed canonical signed-fact language interface

## Claim and decision boundary

V30 tests whether one frozen language model can normalize supported relational clauses into a
canonical fact containing a predicate, ordered arguments, and an epistemic truth status of
`true`, `false`, or `unknown`. The primary extractor never sees a candidate fact. It receives only
the declared typed ontology, entity inventory, and evidence clause; deterministic code performs
any later alignment to a public candidate universe.

This is a fresh, sealed language-only development study. Passing does not establish general
world-model learning, ontology induction, open-vocabulary entity discovery, causal discovery,
sequential dynamics, planning, or a final relational result. V28 remains the selected
episode-level program-inference backend within the existing one-step Boolean ontology.

## Frozen benchmark construction

The generator creates 35 explicitly registered surface families: three fit, one calibration, and
three sealed evaluation families for each of `affirmative_gold`, `negated_opposite`,
`contrastive_both`, `double_negation`, and `explicit_unknown`. A surface family is a syntactic
construction, not a lexical substitution. Every evaluation operator and semantic signature is
supported by disjoint fit constructions; completely absent operators are not used as gates.

Each family contains two typed base scenes and four controlled variants. Direct clean scenes carry
the three unary and two directed-relation predicates. Inverse variants preserve the two relation
facts while changing only their linguistic realization. Distractor variants preserve all facts
while adding an irrelevant clause. Argument-reversal variants replace `linked(a,b)` with
`linked(b,a)`. Entity counts span three, four, and five. Cross-family identifiers register
affirmative/negated-opposite equivalence and known-false/explicit-unknown contrasts.

The evaluation unit is the surface family. Exact evidence strings and exact primary prompts may
not cross splits. Agent input may contain no target, candidate fact, truth label, semantic
operator, pair role, or generator provenance. Before model access, the audit must reproduce every
surface from its private canonical target, verify all controlled pairs, confirm split and semantic
support, and hash the corpus, generator, prompts, evaluator, gates, and this preregistration.

## Frozen comparison ladder

1. **Locked V26 baseline.** With the oracle atom supplied solely to isolate truth normalization,
   reuse the V26 A/B/C prompt style, full-depth frozen decoder, and direct float32 label logits.
   It is not credited with predicate or argument discovery.
2. **Primary canonical extractor.** Ask four candidate-independent constrained questions for
   predicate, argument 1, argument 2 (including `N/A` for unary predicates), and truth status.
   Fixed A–F label logits are projected in float32 and selected by argmax with a fixed tie order.
   No prompt, option order, threshold, field weight, or model is selected from fit, calibration,
   or evaluation results.
3. **Candidate-NLI diagnostic.** Only if the primary fails any sealed language gate, use a second
   preregistered explicit-semantics prompt with the oracle atom and fixed A/B/C truth labels. This
   determines whether the frozen model retains the correct relation under a conditional interface.
4. **Oracle ceiling.** Generator provenance is round-tripped deterministically. This validates the
   benchmark only and is never reported as a language model.

All methods use `mlx-community/Qwen3.5-4B-4bit` at revision
`0e7ffd5c629ef7719d4cbc04069232580bfa9d9c`. Fit and calibration are report-only because no model,
head, prompt, threshold, or hyperparameter is fitted or selected.

## Metrics and non-compensatory gates

Report predicate, first-argument, relation-order, truth-status, exact signed-fact, and exact-scene
accuracy. Also report every truth class, semantic operator, surface family, predicate kind, entity
count, sentence-length stratum, and scene variant. Controlled pairs report both predictions exact,
not merely equal, so two matching mistakes cannot pass an invariance gate.

The primary must meet every registered aggregate, worst-operator, worst-surface-family,
worst-truth-class, exact-scene, distractor, inverse, argument-reversal,
affirmative/negated-opposite, and false/unknown gate. Evaluation cannot compensate one failed
family with easy affirmative cases.

## Conditional V28 reintegration

Only a complete primary language-gate pass authorizes one reintegration replay on the already
exposed V22r2 development corpus. The extractor remains candidate-independent. For each scene,
deterministic maximum-weight one-to-one matching aligns its frozen field log probabilities to the
public positive atom statements; truth status is unchanged. The ontology, V22 DSL, executor,
support outcomes, program inventory, V28 marginal-program-MAP function, gates, and tie rules are
otherwise unchanged.

The replay reports all four support/query oracle decompositions, exact support graphs, complete
episodes, target-program top-1, target retention after every support prefix, empty version spaces,
and comparison with V28. It occurs once and cannot be tuned.

## LoRA eligibility

This protocol never trains an adapter. A narrowly scoped signed-fact grounding adapter becomes
eligible for a separately preregistered study only if the structural audit passes; the primary and
conditional oracle-atom NLI diagnostic both fail the registered truth requirement on at least two
independent evaluation surface families of a shared supported operator; the correct truth is not
reliably retained in the top two alternatives; and no evaluation result selected a prompt,
threshold, field weight, or model. Otherwise the decision is to repair the interface or benchmark
coverage without weight training.
