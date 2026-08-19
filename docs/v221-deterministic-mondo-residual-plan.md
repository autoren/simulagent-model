# V221 deterministic Mondo residual plan

V221 measures what remains after the complete allowed deterministic stack on the verified V220 development population.
It is not a model benchmark and does not open the V220 protected partition.

Before a development JSONL body is loaded, the design audit derives a group-disjoint role manifest solely from the
1,621 development group IDs already present in the V220 population manifest. Groups are ordered by the SHA-256 of
`V221_ROLE|group_id`; indices divisible by five are evaluation and all others calibration. The expected split is 1,296
calibration and 325 evaluation groups. Methods, budgets, controller behavior, safety gates, and the residual threshold
are frozen before either role is evaluated.

The deterministic catalog is reconstructed from only the exact V220 older/newer OBO payloads and the frozen V220
asserted-state semantics. It indexes normalized names and synonyms, exact state classes, stable IDs, and asserted
`replaced_by`/`consider` family links. It performs no remote import resolution and asserts no OWL-inferred equivalence.

Four nested methods are evaluated at candidate budgets `1`, `4`, `8`, and `16`: normalized exact lookup; atomic exact
family expansion; fixed token/character retrieval with family expansion; and the final fail-closed method. Families are
atomic: a budget can include the entire applicable family state set or none of it. Exact equivalence-class identity
collapses duplicates. Retrieval-only, overflowed, absent, or lifecycle-conflicting evidence cannot authorize a
singleton action.

The primary budget is eight classes. Results are reported by role, method, budget, evidence mode, and event stratum for
oracle-class recall, full version-space retention, candidate size, ambiguous singletons, unsafe singleton collapse,
exact decision, and decision regret. Evaluation records cannot change any method, score, threshold, budget, or
controller rule.

A model-eligible residual requires at least 12 evaluation groups at the primary budget where the final method still
misses a decision-relevant frozen oracle class, with zero unsafe singleton collapse and exact fail-closed conflict
handling. Fewer than 12 closes model escalation for this retrospective task. Twelve or more authorizes only a separate
local-model design; it does not authorize a model run.

Protected JSONL paths and hashes may be verified, but their contents may not be loaded. No network, model, training,
registration, mutation, action, or execution is permitted. Historical ontology evidence remains distinct from claims
about a new speaker's intended meaning.

