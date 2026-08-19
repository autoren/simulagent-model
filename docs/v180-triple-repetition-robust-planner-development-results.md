# V180 triple-repetition robust-planner development results

## Verdict

V180 is a **strong positive development result**. The fixed triple-repetition code remains cost-effective after charging
all three raw inspections in every measurement block.

The exact robust certification planner has mean routed risk `33856/20475 = 1.6535286935`, below immediate deferral at
`2`. It completes a trusted route with probability `2/3`, improves all 135 development states over immediate deferral,
and is pointwise no worse than every operational control. This is development evidence on reused V176 states, not fresh
confirmation.

## Complete results

| Policy | Mean routed risk | Expected blocks | Expected raw inspections | Trusted completion |
|---|---:|---:|---:|---:|
| Immediate defer | 2.000000 | 0 | 0 | 0 |
| V175 clean policy, repriced at triple cost | 1.653529 | 3.289540 | 9.868620 | 0.666667 |
| Greedy information gain | 1.729752 | 3.543617 | 10.630851 | 0.666667 |
| Optimal fixed open loop | 1.866667 | 4 | 12 | 0.666667 |
| Random block order | 1.705345 | 3.462262 | 10.386786 | 0.666667 |
| Exact robust adaptive | **1.653529** | **3.289540** | **9.868620** | **0.666667** |
| Target-informed certificate oracle | 1.596882 | 3.100719 | 9.302157 | 0.666667 |

The target-informed row is a non-operational lower-bound comparator. It has no routing or mutation authority.

## What the result establishes

- All 135 eligible V176 states and all 2,160 targets were scored under all seven frozen policies: 15,120 exact target-policy
  scores with no selection or subsampling.
- The exact dynamic-programming root risk reconstructed at rate `1.0`.
- Triple-majority routes were invariant across every admissible zero-or-one corruption scenario, by the V179 equivalence
  proof reused here.
- False trusted routing and provisional sandbox entry were both exactly zero.
- The V171 sandbox reproduced exactly, preserved every invariant, and verified provenance and restart state on all 1,908
  simulated trusted transactions.
- Planner commit authorization, models, APIs, real sensors or services, ontology registration, trusted real-state mutation,
  external effects, and execution all remained zero.

## Important boundary

The exact robust policy and the old clean-cost policy choose the same effective query behavior after the latter is
repriced at the true block cost: both have risk `1.6535286935` and expected block count `3.2895400895`. Thus V180 does
not show that corruption changes which query should be asked on this development population. It shows that the V175
query strategy survives the fixed error-correcting wrapper and remains worthwhile at three times the inspection cost.

The oracle gap is small but nonzero: `1.653529 - 1.596882 = 0.056646`. This leaves limited room for a target-blind
planner improvement without changing the frozen observation code or authority boundary.

## Decision

Freeze V180 as a strong positive triple-repetition robust-planner development result. Do not tune the cost, horizon,
repetition code, decoder, unanimity gate, population, or thresholds. The next admissible test is a separately frozen,
exact-context-disjoint population evaluated with the unchanged robust mechanism. Language models and real observation
channels remain out of scope until that confirmation is resolved.
