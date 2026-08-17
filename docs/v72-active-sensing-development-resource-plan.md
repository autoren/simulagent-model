# V72 RockSample development exporter and resource plan

Implement the pinned 2×2 `RockSample.jl` blueprint as a deterministic finite kernel, preserving the source state order, seven global actions, three observations, deterministic transitions, distance-dependent check channel, and source rewards. The only project-authored transformation is the already-frozen binary observation-label codebook with a `0.2` uniform good/bad noise-floor mixture for check actions.

This stage may test normalization, state enumeration, transition, reward, observation, common support, initial belief, source hashes, and analytic resource bounds. It must not call a planner on the selected blueprint, run the source simulator, or compute a policy value, optimal action, regret, or EIG.

The exporter is resource-feasible only if it has at most 64 states, 8 actions, and 3 observations; its three dense arrays use at most 100,000 bytes; and the structural upper bound for recursive Bellman nodes at the locked horizon is at most 10,000. Passing this stage authorizes writing and auditing an evaluator, not candidate outcomes.
