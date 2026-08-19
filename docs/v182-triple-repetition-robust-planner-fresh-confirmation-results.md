# V182 triple-repetition robust-planner fresh-confirmation results

## Verdict

V182 is a **strong fresh confirmation** of the fixed V180 triple-repetition robust mechanism on the separately frozen
V181 five-constraint population.

The unchanged exact robust planner has mean routed risk `793/550 = 1.4418181818`, compared with `2` for immediate
deferral. Trusted completion is `2/3`; all 66 states strictly improve over immediate deferral; and the exact policy is
pointwise no worse than every operational control. This confirms context generalization within the fixed ontology under
the declared one-corruption code. It is not unseen-concept, language, learned-observation, or real-sensor evidence.

## Complete results

| Policy | Mean routed risk | Expected blocks | Expected raw inspections | Trusted completion |
|---|---:|---:|---:|---:|
| Immediate defer | 2.000000 | 0 | 0 | 0 |
| V175 clean policy, repriced at triple cost | 1.441818 | 2.583838 | 7.751515 | 0.666667 |
| Greedy information gain | 1.502424 | 2.785859 | 8.357576 | 0.666667 |
| Optimal fixed open loop | 1.566667 | 3 | 9 | 0.666667 |
| Random block order | 1.482222 | 2.718519 | 8.155556 | 0.666667 |
| Exact robust adaptive | **1.441818** | **2.583838** | **7.751515** | **0.666667** |
| Target-informed certificate oracle | 1.407879 | 2.470707 | 7.412121 | 0.666667 |

The target-informed oracle is non-operational. Its risk gap from the exact target-blind policy is `0.033939`, leaving
limited room for improvement without changing the frozen ontology or observation design.

## Reconstruction and safety

- Every one of 66 states and 528 targets was scored under all seven policies: 3,696 exact target-policy scores with no
  selection or subsampling.
- Exact dynamic-programming risk reconstruction, population coverage, prior normalization, and corruption-route
  invariance all equal `1.0`.
- All 528 target certificates are valid. Minimal certificates require one block for 240 targets, two for 144, and three
  for 144—respectively 3, 6, and 9 raw inspections.
- False trusted routing and provisional sandbox entry are exactly zero.
- The V171 sandbox reproduced exactly, preserved every invariant, and verified provenance and restart state across 828
  simulated trusted transactions.
- Planner commit authorization, models, APIs, real sensors and services, registration, trusted real-state mutation,
  external effects, and execution all remained zero.

## Interpretation

The exact robust policy again matches the old clean policy after repricing that policy at the true triple-inspection
cost. Together, V180 and V182 support a stable result: fixed majority-code redundancy increases observation cost but
does not alter the optimal query tree on either population, and the resulting trusted completion is still worth that
cost under the frozen routed loss.

The confirmation population begins with five trusted constraints rather than four, so fewer blocks are needed on
average and mean risk is lower than V180's `1.653529`. That difference is a population-structure effect, not evidence
that the mechanism was changed or improved.

## Decision

Freeze V182 as a strong fresh confirmation and close the fixed-ontology, one-corruption triple-repetition branch. Do
not rerun, tune, change the code, cost, horizon, decoder, gate, population, controls, or thresholds. Any additional
robustness work must pose a genuinely new question—for example multiple corruptions, non-binary observations, or a
learned untrusted observation channel—and must begin with a new structural population and lock.
