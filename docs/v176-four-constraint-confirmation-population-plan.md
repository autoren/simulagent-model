# V176 four-constraint confirmation population plan

## Purpose

V176 creates the fresh population for confirming V175. It computes no planner, policy, routed-risk, certificate,
sandbox, model, or execution result. Membership is frozen before the unchanged V175 mechanism can be scored.

## Complete frame and eligibility

Enumerate every choice of four distinct valuations from eight and every binary outcome assignment: `C(8,4) * 2^4
= 1,120` source states. Retain all source states. Exact truth-table filtering leaves 16 candidates per state.

A state is confirmation-eligible exactly when its version space contains alias, composition, and
`provisional_primitive`. This is structural, not outcome-based: the unchanged V167 prior used by V175 requires all
three classes. Every candidate in every eligible state becomes a target and receives one third of prior mass per
class, uniform within class.

## Freshness

The exact target-context signature is the canonical initial constraint signature plus the target candidate's complete
eight-valuation truth table. Every V176 target-context signature must be absent from V172. This establishes new exact
evidence contexts.

The fixed candidate ontology is intentionally reused. Candidate IDs are therefore not required to be disjoint, and
V176 must not be described as unseen-concept evidence.

## Boundary

V176 stores only source states, exact version spaces, target identities, structural class metadata, weights, and
freshness hashes. A passing population audit authorizes a separate confirmation design, not an immediate run. Models,
APIs, registration, trusted state, real services, effects, and execution remain at zero.
