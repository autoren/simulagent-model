# V73 structural dominance and discriminability plan

## Objective

Before any optimal contingent planner runs, V73 must show that its calibration beacon cannot reproduce V72's reward shortcut and that the sensor can change the later maintenance decision. The frozen adapter has three target damage states, four actions, four observations, a five-action horizon, and two shared-support label codebooks.

## Noncompensatory audit

The audit first binds the exported matrices to IMPRL component 4. It then checks that calibration applies the source `do_nothing` transition, has no state-control effect, costs inspection plus mobilisation, and has no harvest or positive reward. Every immediate state-action reward is enumerated; any strictly positive reward rejects the design.

The sensor audit independently computes calibration mutual information, healthy-versus-degraded inspection total variation, and a one-step maintenance threshold. With source replacement plus mobilisation cost `94`, projected failure cost `270`, and prior healthy/degraded mass `0.6/0.4`, the two paired calibration/target label patterns must place the degraded posterior on opposite sides of the frozen replacement threshold. This is a structural proxy, not an optimal-policy result.

Finally, the audit enumerates all `4^5` open-loop action sequences and evaluates one fixed adaptive policy: calibrate, inspect, replace on label 2 or label mismatch, otherwise do nothing, followed by do-nothing actions. Its normalized advantage over the best open loop must be at least `0.005`. This pre-outcome dominance test is intentionally allowed; exact Bayes-adaptive, MAP, posterior-sampling, and myopic calls remain forbidden.

A full pass authorizes only evaluator implementation and locking under the separately preregistered development plan. Any failure closes V73 before planner outcomes. No parameter or gate may be changed after this plan is locked.
