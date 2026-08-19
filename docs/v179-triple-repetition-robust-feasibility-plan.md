# V179 triple-repetition robust-feasibility plan

## Fixed code

Each selected valuation is measured three times as an indivisible shadow block. At most one raw result may be corrupted
over the whole episode. Strict majority decoding therefore returns the target's true bit for every completed block.
The repetition length is fixed from coding distance before the census, not selected from target outcomes.

## Structural proofs and census

For every target, enumerate the no-flip history and every possible single raw flip over all four remaining measurement
blocks. Verify both:

1. every decoded block equals the target truth bit; and
2. the raw set-membership version space with mismatch budget one equals the clean version space conditioned on decoded
   bits.

Then compute exact minimal class certificates in completed blocks and raw inspection counts. Because decoded bits are
exact under the corruption model, compute target-blind worst-case trusted-completion opportunity at block horizons zero
through four using the frozen class-balanced prior.

## Boundary

V176 is declared development for the repeated-measurement mechanism. V179 scores no raw-inspection cost, routed loss,
planner comparator, or sandbox transaction. Positive feasibility authorizes a separate robust-planner design using a
prospectively frozen block cost. It does not authorize an immediate run, a different repetition code, weaker routing,
models, APIs, real sensing, registration, services, state mutation, effects, or execution.
