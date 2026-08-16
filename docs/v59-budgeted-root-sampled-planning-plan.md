# V59 preregistration: budgeted root-sampled Bayes-adaptive planning

V59 advances the verified-agent core after V56 while leaving the unavailable V58 human-language branch deferred. It changes only the planning search. The V53r2 exact joint belief, V55r1 planning-specific mechanic registry, symbolic executor, observations, rewards, action costs, and complete five-action set remain fixed.

## Question and claim boundary

The candidate is a root-sampled UCT search over action–observation histories. Every simulation samples one complete joint belief atom at the root and keeps its program, parameter, world, and hidden queue trajectory coherent for the full rollout. Tree actions depend only on public action–observation history. Rollout action sampling is independent of hidden latent identity and state.

The study asks whether finite search budgets recover exact Bayes-adaptive root decisions at horizon three and whether observation-contingent search is useful at horizons five and seven. It does not claim exact long-horizon optimality, approximate-inference correctness, formal safety, human-language robustness, or unbounded planning.

## Reference and scale strata

The sealed population contains 24 tasks: eight each at horizons three, five, and seven. The horizon-three stratum is scored against the frozen exact V55 dynamic program. Exact root regret evaluates the candidate-selected action using the exact root action value with exact continuation. Horizons five and seven are scaling strata because the exact dynamic program is already impractical at horizon four for a typical 2,056-atom belief.

Each candidate tree is evaluated with 2,048 independent posterior-predictive episodes. The main scale control uses the same root samples, UCT rule, rollout, simulation budget, and tie-breaking but merges all observations. It therefore searches a history-blind open-loop tree under an equal simulator-call budget. Common random numbers pair candidate and control policy-return estimates.

## Budgets, controls, and integrity

Search budgets are 64, 256, and 1,024 simulations, with three independent preregistered replicates. Candidate actions are never pruned. Unvisited deployment histories use a deterministic history-hash action that has no latent access.

Implementation controls must detect nonpersistent root latents, observation misrouting, action-cost omission, budget miscounting, and any rollout action that reads hidden program or state. Candidate evaluation receives only the separately sealed public population file; truth fields remain in a distinct audit file that the evaluator is forbidden to open.

All gates are noncompensatory. A pass qualifies only calibrated bounded search and productive observation contingency on this frozen finite domain. V58 remains deferred, and no simulated record may be represented as human-authored evidence.

