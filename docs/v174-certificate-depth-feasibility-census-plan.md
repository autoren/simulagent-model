# V174 certificate-depth feasibility census plan

## Purpose

V174 measures the exact information geometry behind V173's failure before a new planner horizon is chosen. V172
is explicitly treated as a development population. Every eligible state and every frozen target is included.
This census does not score query cost, routed loss, or a certification-aware policy.

## Certificates

For each target, the census enumerates every subset of the valuation queries not already fixed by the state's
three constraints. A subset is a valid certificate when conditioning on the target's outcomes leaves an exact
version space containing one expressibility class. The minimal depth and lexicographically first minimal subset
are recorded. Target-informed certificate depth is an information upper bound because an operational policy does
not know which target it faces.

Separately, exact dynamic programming computes the maximum expected probability of reaching a unanimously
trusted (`alias` or `composition`) version space at each horizon from zero through five. It uses V167's frozen
class-balanced prior, but no query cost. This policy may condition future queries on observed outcomes but never
on the hidden target. Its value must be monotone and cannot exceed the target-informed certificate upper bound.

## Decision boundary

If the census reconstructs exactly, V175 may prospectively choose a horizon using only the frozen structural
curve. It may not alter V173 or claim that a longer horizon repaired it. V175 will be a new mechanism with a new
routed-risk objective and separate controls.

No language, model, API, sandbox transaction, ontology registration, trusted-state mutation, real service, side
effect, or execution is allowed.
