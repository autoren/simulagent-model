# V65r3 pooled SMC² EIG portability results

Date frozen: 2026-08-16

## Outcome

V65r3 passed every original V65/V65r1 noncompensatory gate and the added exact-zero identity branch
count check. It authorizes preregistration of the external Bayes-adaptive reward-decision stage. It
does not itself test reward planning or formal policy verification.

The run is a targeted repair result, not an independent replication. V65r1 remains a one-shot
runtime failure, and V65r2 remains rejected before evaluation for an implementation-stage access
violation.

## Primary 509-particle result

The qualifying posterior is the equal-weight mixture of three independent SMC² repeat posteriors,
with 509 outer theta particles per identity and 127 inner state particles. Acquisition uses the
V65r1 Rao–Blackwellized known 11-state conditional while retaining the approximate pooled static
identity-theta weights.

| Metric | V65r3 result | Frozen gate |
|---|---:|---:|
| Mean four-action EIG error | 0.000188 nats | ≤ 0.004 |
| q95 four-action EIG error | 0.000688 nats | ≤ 0.015 |
| Strict exact-optimal membership | 0.9792 | ≥ 0.80 |
| 0.001-nat ε-optimal membership | 1.0000 | ≥ 0.95 |
| Mean exact selection regret | 1.88e-9 nats | ≤ 0.0015 |
| q95 exact selection regret | 2.89e-17 nats | ≤ 0.006 |
| Maximum exact selection regret | 9.04e-8 nats | ≤ 0.02 |
| Mean identity TV | 0.00207 | ≤ 0.05 |
| q95 identity TV | 0.00538 | ≤ 0.15 |
| Mean theta Wasserstein-1 | 0.00180 | ≤ 0.04 |
| q95 theta Wasserstein-1 | 0.00300 | ≤ 0.12 |
| Mean joint identity-theta TV | 0.04237 | ≤ 0.10 |
| q95 joint identity-theta TV | 0.05143 | ≤ 0.25 |
| Mean state TV | 0.00115 | ≤ 0.08 |
| q95 state TV | 0.00344 | ≤ 0.20 |
| Mean candidate predictive TV | 0.000759 | ≤ 0.05 |
| q95 candidate predictive TV | 0.00225 | ≤ 0.15 |

The selected pooled action differed from the strict exact-optimal set on one of 48 records, but its
exact regret was only `9.04e-8` nats and it remained ε-optimal. The primary selected-action counts
were `n=23`, `e=4`, `s=18`, and `w=3`.

## Scaling and repeat diagnostics

Mean four-action EIG error decreased from `0.000736` at budget 31 to `0.000385` at budget 127 and
`0.000188` at budget 509. Mean joint identity-theta TV decreased from `0.1669` to `0.0791` to
`0.0424`. The qualifying 509-particle result was better than the low-budget result on both mean EIG
error and mean selection regret, satisfying both registered scaling gates.

At budget 509, individual-repeat strict membership rates were `0.9792`, `0.9583`, and `0.9792`; all
three ε-optimal rates were 1.0. Repeat selections disagreed on 10.4% of records, but pooling reduced
mean exact regret to effectively zero. These repeat metrics remain diagnostics, not substitutes for
the pooled gates.

## Controls, repair, and one-shot integrity

Seven of nine registered controls were detected or dominated, above the minimum of six. The
pool-after-scoring and first-repeat-only controls were not dominated on this subset; MAP identity,
theta mean, equal identity evidence, particle-state plug-in prediction, state-as-target, shared
streams, and outcome leakage were detected or materially worse.

The repaired exact-zero identity branch appeared exactly nine times: one sealed history across three
budgets and three repeats. Each contributed log evidence `-Infinity`, zero posterior mass, and no
atoms. No positive-support particle extinction occurred.

The atomic attempt marker consumed exactly one V65r3 attempt. The evaluator wrote 144 complete
record-budget rows and 432 repeat cells, no failure artifact, no random-stream collision, and no
candidate omission or tie-break violation. It loaded no V64 audit/result record or truth field and
used no human record, model forward pass, or adapter training. An independent outcome auditor
re-aggregated the raw cells and reproduced every original gate and summary.

## Claim boundary and next step

The evidence now supports one-step active-acquisition portability for the pooled SMC² static
posterior combined with an exact known-state conditional on this externally anchored,
project-authored actuator family. It does not qualify the pure nested particle-state predictive and
does not establish sequential Bayes-adaptive reward quality.

The next stage should preregister bounded-horizon reward decisions that compare:

1. exact Bayes-adaptive belief-state planning;
2. pooled-SMC² Bayes-adaptive planning;
3. the true-model oracle;
4. a MAP-model certainty-equivalent planner; and
5. a valid posterior-mixture control that resamples a static model once per simulated episode rather
   than incorrectly averaging transition matrices at every step.

Only after that decision stage passes should the selected bounded policies be independently verified.
