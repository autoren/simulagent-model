# V22 development protocol: typed relational state

## Question and scope

V22 asks whether the successful modular architecture can lift from named Boolean slots to typed,
variable-size relational structures. The primary development question is whether an executable
schema over variables transfers across entity bindings, graph topology, relation orientation, and
entity count under supported language.

The study remains deterministic and one-step. It does not introduce learned ontologies, persistent
next-state mutation, stochastic effects, long-horizon planning, active intervention selection, or
model-weight adaptation. V22 development artifacts cannot be presented as a final evaluation.

## State and epistemic semantics

A complete state is a typed entity set plus a truth value for every well-typed unary and directed
binary atom. Omitted atoms are forbidden. A complete false atom is known false; it is not an
unobserved fact.

An epistemic state assigns every well-typed atom one of `{false}`, `{true}`, or `{false, true}`.
The last set means unobserved or unresolved. Query execution ranges over every compatible complete
state and returns the union of possible visible outcomes. Serialization omission never means
unknown and never invokes an open-world assumption.

Every action is `inspect_pair(actor, target)` with two distinct `unit` bindings. Programs refer to
these parameters and may bind at most one additional typed variable. Entity identities have no
semantic role: consistent renaming and graph isomorphism must transform predictions equivariantly.

## Lifted DSL

The finite DSL contains typed unary atoms, directed relational atoms, `not`, `and`, `or`, `xor`,
and one bounded existential variable. The first catalog represents four sources of dependence:

1. unary selection on action arguments;
2. direct, oriented relation conditions;
3. two-hop paths through a bound unit; and
4. permutation-invariant existential aggregation over related units or hubs.

Expressions are canonicalized under bound-variable renaming and commutative Boolean operations;
relation arguments remain ordered. Behavioral equivalence is checked over every permitted entity
layout and action binding using counterexample search over only the grounded atoms relevant to the
two expressions. Exhaustive enumeration of all complete graphs is neither claimed nor required.

## Identification and entity-count extrapolation

Support traces contain only two- and three-entity states. A target is eligible only if every other
candidate program has a concrete distinguishing support-domain counterexample. Greedy support
selection operates over those counterexamples and must leave exactly the target behavior.

This prevents a four-entity query from testing arbitrary prior choice between programs that were
indistinguishable at smaller counts. Four-entity queries can nevertheless add witnesses, paths, or
aggregate members and therefore change relational semantics.

All current episodes are behavior-identifiable. A later ambiguous-support condition must return a
union over the surviving lifted version space and will be registered separately.

## Development axes and metamorphic controls

The generator includes binding recombination, relation orientation, graph topology, entity-count
extrapolation, partial observation, distractor invariance, and permutation equivariance. Hard
metamorphic checks rename entities, reorder serialization, preserve graph isomorphism, reverse
relation arguments, and add an inert entity. Graph and program identities use canonical hashes, not
raw entity names or text.

Relational language signatures explicitly record predicate kind, ordered arguments, truth status,
surface operator, and direct or inverse realization. Existing V14 polarity operators do not count
as support for unseen argument orientation or quantifier semantics.

## Oracle-first decision rule

The first implementation authorizes no model access. It must establish:

- exact target recovery in the finite lifted candidate class;
- exact oracle query execution, including unresolved facts;
- exact permutation and distractor metamorphic behavior;
- correct relation-orientation signatures;
- structural disjointness of fit/evaluation programs and support/query graphs;
- zero target or oracle-state leakage into agent inputs; and
- tractable candidate, runtime, and memory scaling through four entities and two output bits.

If the oracle system fails, revise the DSL, generator, equivalence checker, or support policy. If it
passes, V22 may proceed to a separately registered relational-language grounding development stage.
No final V22 construction is authorized until all model and challenger choices are later frozen.
