# Research roadmap after V178

## Boundary established

V178 proves that single-pass inspections are structurally insufficient under one possible corrupted outcome. No alias
or composition target has a robust class certificate, even with target identity available to the oracle. Consequently,
no target-blind policy can achieve trusted completion. The conservative gate is safe; the observation code has no
error-correcting distance.

Do not rerun V178, weaken unanimity, add posterior thresholds, or tune the population.

## Next active track: fixed triple-repetition feasibility

Use a prospectively fixed repetition code:

- selecting one valuation executes three identical shadow inspections as one indivisible measurement block;
- at most one raw inspection outcome may be corrupted globally;
- decode each completed block by majority;
- expose only the decoded bit to the version-space planner;
- route only when the resulting robust version space is unanimously alias or composition;
- defer mixed and unanimous provisional sets.

Three repeats are selected from the corruption model, not from V178 outcomes: a binary repetition code of length three
has distance three and corrects one bit error. Two repeats cannot deterministically resolve a one-error disagreement.

### Recommended sequence

1. **V179 triple-repetition feasibility census.** Prove exhaustively that majority decoding is exact for every target,
   queried valuation, and admissible global flip. Show that the raw robust version space equals the clean version space
   conditioned on decoded bits. Compute minimal robust certificates in measurement blocks and raw inspections, plus
   target-blind trusted-completion opportunity by block horizon. Score no cost, routed risk, or sandbox.
2. **V180 robust planner development only if V179 is positive.** Use query-block cost `3 * 0.1 = 0.3`, retain V175's
   routed loss and authority gate, and compare against immediate deferral, the clean policy naively priced/executed,
   greedy, random, fixed open-loop, and a non-operational target-informed robust oracle. Report both block and raw
   inspection counts.
3. **Fresh confirmation only after positive development.** Freeze another exact-context-disjoint population before
   scoring the unchanged robust mechanism.

## Other tracks

The clean V175/V177 mechanism remains closed and strongly confirmed. V171 remains the sandbox contract. Language and
models remain dormant: coding-theoretic observation robustness should be resolved before adding an untrusted semantic
channel. Registration, real services, real-state mutation, effects, and execution remain zero.
