# V62 post-hoc terminal-residual diagnostic

This is a labeled post-hoc diagnostic. It does not revise V62's immutable failed qualification.

- All `4` reachable nodes whose old residual exceeded `1e-10` had belief support entirely on absorbing terminal states.
- There were `0` nonterminal residual failures.
- The planner returned zero value and zero action values at every terminal-support node, with `0` violations.
- The pinned POBAX runtime declares an all-action absorbing successor terminal, and the frozen rollout excludes done episodes before the next reward-bearing step.
- The original per-cell residuals reproduced exactly without another rollout or candidate evaluation.

The old residual checker special-cased horizon zero but not terminal belief support. It therefore recomposed a counterfactual action reward after the runtime and planner had already stopped. The proper next step is a separately preregistered measurement repair that adds the missing terminal-support base case, mutation-tests that rule, and rescores only the immutable reachable belief nodes. V62 itself remains a 31/32-gate failure.
