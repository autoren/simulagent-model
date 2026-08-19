# V178 one-corruption robust-certificate feasibility results

## Verdict

V178 is a valid and decisive structural negative result: single-pass inspection cannot robustly certify either trusted
class when one subsequent observation may be corrupted.

Across all 135 declared development states, 2,160 targets, and 10,800 target/corruption scenarios, target-blind
worst-case trusted completion was exactly zero at every horizon from zero through all four remaining valuations. The
target-informed trusted upper bound was also exactly zero, so this is not a policy-search failure.

## Why it fails

Four trusted initial constraints leave 16 candidates corresponding to the 16 possible patterns on the remaining four
valuations. After single-pass inspection, allowing one disagreement means the robust version space contains the
observed pattern and its Hamming-distance-one neighbors. For every alias and every composition target in this
population, that robust neighborhood includes another expressibility class. Unanimous trusted routing is therefore
impossible even when the target is known to the oracle.

The exact class results were:

- alias: 0 of 138 targets certifiable;
- composition: 0 of 180 targets certifiable;
- provisional: 216 of 1,842 targets class-certifiable, all at depth four;
- overall: 216 certifiable and 1,944 uncertifiable targets.

Provisional certificates do not produce trusted completion because provisional candidates remain outside the sandbox.

## Safety and integrity

All structural gates passed. The true target stayed in the robust version space in every admissible scenario, all
reported witnesses were valid and minimal, adaptive opportunity was monotone and bounded by the target-informed upper
bound, and false trusted routing remained zero.

V178 computed no query cost, routed-risk score, planner comparison, or sandbox transaction. There was no model/API use,
registration, real-state mutation, service call, side effect, or execution.

## Correct successor

Do not weaken unanimity, add a posterior threshold, rerun V178, or tune the population. The obstruction is a lack of
error-correcting observations.

The justified next study is a separately preregistered repeated-measurement feasibility design. With at most one
corrupted observation globally, repeating each selected valuation three times and decoding by majority is the simplest
fixed construction that recovers its true bit deterministically. The next census should compare prospectively fixed
redundancy designs structurally before any robust planner cost or routed-risk evaluation.
