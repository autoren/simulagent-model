# V178 one-corruption robust-certificate feasibility plan

## Question

Can the confirmed certification architecture still produce a trusted route if at most one new inspection outcome may
be wrong, while keeping deterministic unanimity rather than replacing it with a probability threshold?

## Robust version space

The four initial V176 constraints remain trusted. For any subsequent history, retain every candidate in the initial
16-candidate version space whose predictions disagree with the observed inspection bits at no more than one queried
valuation. The true candidate is therefore retained under every admissible no-flip or single-flip history.

Route only when all retained candidates are alias or all are composition. Mixed sets defer, and unanimous provisional
sets defer outside the sandbox.

## Exact structural estimands

For every target and every subset of the four remaining valuations, enumerate the no-corruption history and every
history formed by flipping one queried outcome. A target-informed robust certificate exists at a depth only when every
admissible history produces the correct unanimous target class. Retain uncertifiable targets.

Separately enumerate every deterministic target-blind adaptive query tree through horizons zero to four. A trusted
target counts as robustly completed only if the same policy succeeds for that target under no flip and under every
possible single flipped valuation. Maximize class-balanced target mass and break ties by canonical policy-tree order.

## Boundary

V176 is explicitly reused as development for this new corruption mechanism. V178 scores no query cost, routed loss,
planner comparator, or sandbox transaction. A positive result permits a separate robust-planner design. A zero result
permits only a prospectively fixed repeated-measurement feasibility study; it does not permit weakening unanimity,
adding a posterior threshold, or repairing V178 in place. Models, APIs, registration, real state, services, effects,
and execution remain zero.
