# V202 model-free controller decision-sufficiency plan

## Question

Can a trusted clarification controller use the presentation-sensitive normalized proposals in a way that is safe,
low-cost, and robust without treating the lower ranks as a semantic posterior?

V202 uses only normalized V195/V201 model artifacts and matched V194/V200 `CHAR_LAST` predictions. It performs no
language read or model generation.

## Fixed policies

- `SINGLE_PRESENTATION_TOP1_FAMILY`: one top contract plus trusted `OTHER`, evaluated across canonical, order-only,
  and opaque-ID presentations. If selected, the fixed implementation is canonical.
- `SINGLE_PRESENTATION_TOP3_FAMILY`: the analogous three-contract question.
- `TOP1_PLURALITY_3X`: plurality of three top-1 contracts; ties use canonical top-1 when available.
- `TOP3_INCLUSION_CONSENSUS_3X`: contracts appearing in at least two of the three top-3 sets, ordered by votes then
  canonical rank then contract ID, capped prospectively at four.

Question cost is `0.10 * ceil(log2(size + 1 for OTHER))`. A miss adds the frozen `0.40` generic clarification cost.
Empty/invalid/insufficient proposals use the target-specific frozen hierarchy. Every exact terminal requires a
trusted answer and the full hypothesis set is retained.

## Gates and selection

Each policy must have robust primary cost at most `0.24`, robust macro cost at most `0.25`, at least `0.01`
worst-condition improvement over its matched `CHAR_LAST` construction, at most `0.10` target-hit disagreement across
presentations, at most `0.05` mean per-record presentation cost range, proposal size at most four, exact target
retention and completion, and zero false terminal decisions.

Selection first considers qualified one-call families only. Three-call policies are eligible only if no one-call
policy qualifies. Within a tier, minimize robust primary cost, then macro cost, proposal size, and policy ID. This
prevents a multi-run ensemble from winning on a tiny human-cost difference while hiding its threefold inference
requirement.

## Boundaries

V202 is a post-V201 development mechanism study and does not change V201's negative result. A selected controller
requires a new fresh or external confirmation population; V198 protected reuse is forbidden. Raw responses, language,
model/API/training access, ontology mutation, services, side effects, action, and execution remain zero.

