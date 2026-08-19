# V173 trusted-only shadow integration results

## Outcome

V173 is a clean safe-but-nonbeneficial integration result. Every integrity and safety gate passed, but every
preregistered benefit threshold failed. The unchanged V167 policies and V171 sandbox contracts reconstructed
exactly across all 132 frozen states, 4,224 target cases, and 29,568 target-policy scores.

The key class-balanced means were:

| Policy | Recommendation risk + query cost | Safe routed risk + query cost | Expected queries | Trusted completion |
|---|---:|---:|---:|---:|
| Oracle comparator | 0.0000 | 0.6667 | 0.0000 | 0.6667 |
| No-query Bayes terminal | 2.0000 | 2.0000 | 0.0000 | 0.0000 |
| Forced MAP, no query | 3.3333 | 2.0000 | 0.0000 | 0.0000 |
| Exact Bayes-adaptive | 1.2031 | 2.1806 | 1.8063 | 0.0000 |
| Greedy information gain | 1.2428 | 2.2000 | 2.0000 | 0.0000 |
| Optimal open-loop pair | 1.2318 | 2.2000 | 2.0000 | 0.0000 |
| Random open-loop pair | 1.5057 | 2.2000 | 2.0000 | 0.0000 |

The exact adaptive policy improved its original recommendation objective, reproducing the V167 mechanism, but it
never produced a version space whose candidates were unanimously `alias` or unanimously `composition`. Its
terminal version spaces were mixed 83.00% of the time and unanimously provisional 17.00% of the time. The
deterministic gate therefore deferred in every operational case. Inspection cost raised routed risk above the
no-query policy in all 132 states; the count of strict statewise improvements was zero.

The non-operational oracle comparator shows that the population contained substantial trusted opportunity: with
the target revealed before applying the same gate, trusted completion was two thirds under the class-balanced
prior and routed risk was two thirds because provisional targets still defer at loss two. The failure is therefore
not an absence of trusted targets. It is an identifiability-and-objective mismatch within the two-query horizon.

## Safety result

The composition failed closed exactly as designed:

- false trusted-route probability was zero for every policy;
- provisional sandbox-entry probability was zero;
- planner commit-authorization count was zero;
- target and policy coverage, prior normalization, and deterministic gate reconstruction were 100%;
- all original V167 recommendation risks reconstructed exactly; and
- every simulated trusted route had exact final state, invariant preservation, authorized mutation, provenance,
  and restart verification.

There were 372 simulated sandbox transactions, exactly the 144 alias plus 228 composition target cases routed by
the non-operational oracle comparator. No operational policy reached the trusted sandbox. No real state or service
was touched.

## Interpretation

V170 established that two adaptive queries can improve a cost-sensitive *classification* decision. V173 shows
that this does not imply two queries can satisfy a much stricter *unanimous certification* rule. The planner was
optimized to choose a low-loss terminal label or defer, not to drive the entire surviving version space into one
trusted class. The safety gate correctly refuses to translate posterior confidence or a MAP label into authority.

This is why the recommendation-risk and routed-risk split matters. Forced MAP recommended unsafe classes badly,
with risk 3.3333, but the gate masked those recommendations by deferring and held routed risk to 2.0. Conversely,
the exact planner made better recommendations but paid for information that was insufficient for certification,
so the composed system became worse than immediate deferral.

## Decision and next direction

Freeze V173 as a safe nonbeneficial boundary result and stop before a confirmation study. Do not weaken the
consensus gate, increase the horizon, change the loss, or tune queries on V172 after seeing this outcome.

The justified successor is a separately developed **certification-aware planner** with its own population and
objective. It should treat trusted consensus, safe provisional deferral, and query cost as the terminal control
problem from the start, include an exact feasibility census over horizon, and compare against V167 unchanged as a
baseline. A horizon change or a less conservative certificate requires prospective design and should be reported
as a new mechanism—not a repair to V173.

No language, local model, API model, training, registration, real service, side effect, or execution is justified
by this result.
