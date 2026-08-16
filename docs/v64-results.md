# V64 results: external multi-action exact expected information gain

**Qualification:** PASS

**Decision:** `authorize_preregistration_of_pooled_three_repeat_SMC2_EIG_stage`

V64 qualified a nontrivial external active-identification benchmark and its exact acquisition
reference. The model is POBAX's pinned nonterminating 4x3 POMDP, augmented with a prospectively
frozen project-authored actuator family: a hidden clockwise-versus-counterclockwise failure identity
and a continuous correct-command probability. POBAX does not supply that uncertainty layer.

## Exact selection benchmark

| Metric | Result |
|---|---:|
| Public histories | 192 |
| Candidate-action comparisons | 768 |
| Maximum candidate/reference EIG error | `5.42e-16` |
| Maximum predictive probability error | `4.44e-16` |
| Optimal-set membership | `1.0` |
| Maximum selected EIG regret | `8.88e-16` |
| Informative-record fraction | `0.8646` |
| Mean exact EIG | `0.03740` nats |
| Mean advantage over uniform random | `0.01425` nats |
| Mean advantage over fixed cycle | `0.01279` nats |
| Dominant-command selection rate | `0.6094` |

All four commands—`n`, `e`, `s`, and `w`—were strictly EIG-optimal on at least one sealed public
history. Exact selections were `n:117`, `s:57`, `e:13`, and `w:5`, so the benchmark does not reduce
to a single predetermined informative action.

## Matched adaptive trajectories

Across 512 sealed paired scenarios, mean posterior KL from the joint identity-theta prior at budget
eight was `0.2561` nats for adaptive exact EIG, `0.1700` for the fixed `n,e,s,w` cycle, and `0.1820`
for uniform random actions. The paired comparisons were:

| Budget-8 difference | Mean | Lower 95% bound |
|---|---:|---:|
| Adaptive EIG minus fixed | `0.08610` | `0.06081` |
| Adaptive EIG minus random | `0.07415` | `0.05025` |

All trajectories completed and every posterior normalized. Rewards were retained only as a
diagnostic of the pinned external environment; no reward objective or planning claim was tested.

## Calibration, controls, and integrity

Adaptive SBC used 256 independent prior-predictive scenarios after eight EIG-selected actions. The
minimum rank chi-square p-value was `0.04215`, maximum absolute rank-bin z was `2.3238`, and maximum
coverage z was `1.25`. These pass the registered calibration gates; the claim is not “perfect
calibration.”

Predictive entropy, current-state information, MAP identity, theta point mass, wrong permutation,
and outcome leakage were all detected or dominated. The candidate read only the 192 public
selection histories. Static truths and streams were used solely by the sealed environment and SBC
rank construction after each action had been selected. There was zero candidate truth access, zero
pre-selection outcome access, zero candidate omission, zero tie-break violation, zero stream
collision, zero human/simulated-human access, zero model forward passes, and zero adapter training.

## Claim boundary and next gate

V64 establishes the exact benchmark and acquisition reference for one externally anchored,
project-authored four-action unknown-dynamics family. It does not test SMC² acquisition, reward
planning, formal verification, language grounding, or general external active identification.

The next authorized action is preregistration—not construction or execution—of a separate stage
that computes EIG from the deployed equal-weight pool of three frozen SMC² repeats and compares its
action-set membership and exact EIG regret across inference budgets. That stage must also report
single-repeat diagnostics and computational cost.
