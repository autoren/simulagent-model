# V61 preregistration: bounded long-horizon approximate-belief policy verification

V60 passed all twenty-three frozen gates: its calibrated SMC-squared posterior remained close to the exact posterior, its horizon-three root decisions were in the exact optimal set, and its longer-horizon observation-contingent policies retained an exact-posterior return advantage over the blind control. V60 nevertheless used Monte Carlo policy evaluation and explicitly made no formal-policy claim. V61 closes that specific gap.

## Exhaustive frozen census

V61 verifies all 72 primary V60 policies: every one of the 24 sealed public tasks, each of the three frozen search replicates, at the primary outer-particle budget of 509 pooled across three repeats. This selection is exhaustive and fixed before reconstruction. It yields 24 policies at each horizon 3, 5, and 7. No audit-truth field is available to reconstruction, compilation, execution, or verification.

Each policy is deterministically reconstructed from the immutable V59 public task, frozen V53r2 inference, V60 seed schedule, and V59/V60 search implementation. Its tree hash, root action, root statistics, search budget, and tree census must match the sealed V60 result. Reconstruction is source binding, not a new planning evaluation: it does not change a policy, compare alternatives, consult truth, or rerun any V60 gate.

## Independent bounded semantics

The policy is executed under the exact 257-node joint posterior, not under the particle approximation used to construct it. A new transition interpreter separately implements typed binding, due-event delivery, pre-action condition evaluation, deterministic effects, exact Bernoulli mass, delayed scheduling, and canonical successor merging. It may not call the formal continuous unit transition. A separate deployment interpreter implements the visit-count/mean/key decision rule and the deterministic public-history fallback without calling V59's deployment or episode evaluator.

The independent recursive evaluator enumerates every stochastic branch through the task horizon for every exact posterior atom. It reports terminal-goal probability and accumulated terminal reward minus action cost. The verified policy remains a mapping from public action-observation history to action; hidden program, theta, world, and queue may affect dynamics but never action selection.

For every reachable compiled source state, a Z3 relation compares the independent transition support to the formal-executor successor support. Additional checks require a complete typed Boolean world, canonical nonoverdue queues, legal actions, exact depth progression, total observation routing through a tree child or the declared fallback, correct terminal labeling, and no nonterminal deadlock.

## External probabilistic verification

The formal executor compiles each frozen policy and the exact posterior into an explicit finite DTMC. A synthetic root samples the full joint atom. Each action transition carries its negative public cost. After the frozen horizon, a staging state is labeled `success` exactly when the public goal holds, receives terminal reward one or zero, and reaches an absorbing `done` state.

Storm 1.13.0 runs as a standalone process and checks termination probability, terminal-goal probability, and expected accumulated reward. Storm must match the independent recursive evaluator to `1e-9`; every model must terminate with probability one and every emitted transition row must normalize.

The stored V60 2,048-episode policy mean is a secondary source cross-check. For 72 simultaneous comparisons at familywise alpha 0.01, each exact return must fall within the horizon-specific Hoeffding radius `range * sqrt(log(2*72/0.01)/(2*2048))`, with reward ranges 1.03, 1.05, and 1.07 for horizons 3, 5, and 7. A discrepancy is reported; V60 is never rerun or replaced.

## Controls, sealing, and decision rule

Before source policies are accessible, six analytic fixtures test deterministic, stochastic, delayed, contingent, reward, and fallback behavior. Ten mutations target binding, queue delivery, probability direction, deployment choice, fallback seed, observation routing, stochastic support, terminal labels, action costs, and root mass. All fixtures and mutations must pass.

Only after the implementation audit is frozen may the 72 policy/model bundles be built. A manifest binds every file to its source V60 cell and is sealed before the candidate runner can access it. Exactly one sealed verification attempt is allowed. All twenty-seven gates are noncompensatory.

A pass qualifies bounded exact-posterior execution verification for these 72 policies through horizon seven. It does not prove the UCT search algorithm correct or optimal, establish a worst-case or parameter-uniform safety property, extend beyond the frozen symbolic domain, or supply missing human-authored language evidence.
