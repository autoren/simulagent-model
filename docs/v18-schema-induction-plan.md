# V18 development plan: executable transition-schema induction

## Decision

V18 tests whether counterfactual transition traces are sufficient to recover an executable
action model. It does not run LoRA and it does not reuse V17. V17 remains a permanently exposed
evaluation whose positive result is treated only as an oracle-schema language-grounding ceiling.

The first V18 condition deliberately supplies oracle-grounded Boolean trace values while hiding
the transition program. This isolates schema induction from natural-language grounding. A later
condition may replace those oracle groundings with the already frozen V15 grounding pipeline.

## Episodic contract

Each episode contains:

1. a candidate action and four Boolean determinant concepts;
2. natural-language counterfactual support traces paired with visible transition codes;
3. natural-language queries that may leave one determinant unresolved;
4. a target-only oracle grounding for supports and queries; and
5. a target-only executable program in a bounded typed DSL.

The agent input never contains an action-dependency table, expression tree, relevant-determinant
list, or truth table. The symbolic baseline may use target-side oracle grounding, but it must infer
the program from support traces.

The DSL contains Boolean variables plus `not`, `and`, `or`, and `xor`. A program computes one or
two visible outcome bits. Programs are scored by their complete truth-table behavior, not their
surface syntax, so observationally equivalent expressions receive identical credit.

## Development split axes

- **Known primitive recombination:** familiar operator families in unseen output-component pairs.
- **Structural composition:** held-out nested operator structures.
- **Determinant vocabulary:** familiar structures rendered with a held-out state lexicon.
- **Composition depth:** deeper programs than the training episodes.
- **Outcome invariance:** non-injective mechanics with both outcome-sensitive and
  outcome-invariant unresolved-determinant queries.

These axes are reported independently. They are not blended into a single claim about
compositional generalization.

## Baselines

1. **Empirical lookup:** memorize observed assignments and return the whole outcome vocabulary
   for any query requiring an unseen assignment.
2. **Exact version-space induction:** enumerate the bounded DSL, retain every program consistent
   with support traces, and return the union of their possible outcomes.
3. **Future primary condition:** replace oracle grounding with frozen V15 grounding while keeping
   exact schema search and execution unchanged.
4. **Future neural proposal condition:** use a learned model only to rank candidate program
   components; deterministic execution and verification remain authoritative.

## Metrics

- full-table behavioral equivalence;
- relevant-determinant exact match and set F1;
- possible-transition-set exact match;
- balanced identifiability accuracy;
- outcome-sensitive and outcome-invariant unresolved-query accuracy;
- worst development-axis accuracy; and
- prefix support curves measuring how many traces are needed.

## Gates and stop rule

The exact inducer must recover every generated target behavior and answer every full-support query
exactly. Every development axis must contain both identifiable and ambiguous queries, and the
outcome-invariance axis must contain both sensitive and invariant unknowns. The lookup baseline
must remain imperfect, demonstrating that the corpus is not merely a memorization check.

LoRA is ineligible until the corpus and exact baseline pass these gates. If exact enumeration is
already tractable, neural work should target language grounding or candidate ranking—not replace
the executor. A fresh final mechanic may be constructed only after all development choices are
frozen; V17 cannot serve that role again.

## Literature basis

The design combines execution decomposition from ExeDec, factored dynamics from Schema Networks
and HOWM, and independently controlled compositional splits motivated by CFQ, COGS, SLOG, and
SETI. IIT or distributed alignment search is reserved for later causal validation after schema
induction succeeds.
