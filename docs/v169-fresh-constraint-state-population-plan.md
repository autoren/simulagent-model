# V169 fresh constraint-state population plan

## Purpose

V169 creates the untouched population needed to test whether V167's development-informed planner mechanism
transfers without changing its prior, losses, costs, horizon, policies, or tie-breaks. It does not score a policy.

## Generator

The finite DSL has eight valuation positions and 256 truth tables. The generator enumerates every choice of two
distinct valuation positions and all four binary outcome assignments, giving 112 exact 64-candidate version
spaces. It reconstructs every constraint signature used by V165's 48 ambiguous records and removes those
signatures. Every remaining state is retained; membership cannot use a planner or policy score.

A state is marked planner-eligible only when its exact version space contains at least one alias, composition,
and provisional-relative-to-the-DSL candidate, because V167's frozen prior assigns one third mass to each class.
Ineligible fresh states remain in the population and are reported rather than silently dropped.

## Gates and boundary

All source, excluded, selected, and eligible counts are automatically audited. Selected signatures must be unique
and have zero overlap with V165; every version space must contain exactly the truth tables satisfying its two
constraints and every class count must sum to 64. At least 48 eligible states are required for the unchanged
planner confirmation.

This is project-authored procedural population evidence, not external or human evidence. Planner scores, models,
APIs, training, registration, services, side effects, actions, and execution remain zero. Passing authorizes only
a separate locked run of the unchanged V167 planner on all eligible states.
