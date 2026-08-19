# V174 certificate-depth feasibility census results

## Outcome

V174 passed every structural feasibility gate over all 132 V172 development states and 4,224 frozen targets. All
minimal certificates were valid and independently minimal, every target was certifiable by full depth, and the
adaptive opportunity curve was monotone and bounded by the target-informed upper bound.

The decisive result is exact:

- every one of the 144 alias targets required all five remaining queries;
- every one of the 228 composition targets required all five remaining queries;
- provisional targets certified much earlier: 77.80% at depth one, 19.38% at depth two, and 2.81% at depth three;
- no target had minimal depth four; and
- the raw target depth counts were 3,024 at depth one, 726 at depth two, 102 at depth three, and 372 at depth five.

Under V167's class-balanced prior, the target-blind optimal adaptive trusted-completion opportunity was exactly
zero at horizons zero through four and exactly two thirds at horizon five. The target-informed trusted upper bound
had the same curve. Thus no adaptive query choice—not merely the V167 policy—could reach a unanimously trusted
version space before observing all five remaining truth-table entries.

## Interpretation

This explains V173 without blaming policy capacity. The finite universe defines `provisional_primitive` as every
truth table not already in the sparse alias or composition families. Until the complete truth table is known, an
unqueried bit can still support a provisional alternative that imitates a trusted candidate on all observed bits.
Trusted membership therefore requires full identification in this DSL.

Provisional targets can often be proven non-trusted quickly, but that knowledge does not change the safe terminal
action: both a mixed version space and a unanimously provisional version space defer. A certification-aware
planner must decide whether continued querying is worth the chance of ultimately reaching trusted completion; it
should not query merely to prove that deferral was appropriate.

## Decision and boundary

Freeze V174 as exact structural development evidence. Prospectively set V175's maximum horizon to five while
retaining query cost 0.1 and the unanimous deterministic gate. Horizon five is not a post-hoc rerun of V173; V175
is a new routed-risk mechanism and must receive its own design lock.

V174 scored no query cost, routed loss, sandbox transaction, or planner comparison. No language, model, API,
training, registration, trusted-state mutation, real service, side effect, or execution occurred.
