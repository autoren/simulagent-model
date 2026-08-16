# V62 results: external classic-POMDP belief and planning transfer

Qualification: **FAIL**

The immutable V62 run passed 31 of 32 preregistered gates. Its only failed gate was the independent Bellman-residual diagnostic: the maximum reported residual was `1` rather than at most `1e-10`. The failure occurred only for `tiger-alt-start` at horizons 5 and 7. V62 is not retroactively treated as a pass.

All root decisions agreed with the independent scalar planner: optimal-set membership was `1` and maximum value error was `8.881784197e-16`. The unchanged POBAX runtime completed all 24 rollout cells, and every 4,096-episode mean was inside its simultaneous familywise 99% bound.

The control diagnostics also passed: Tiger selected `listen` at every registered information-gathering horizon, the minimum T-Maze exact-history advantage over observation-only was `1.08945405`, and the minimum Tiger exact-history advantage over MAP collapse was `47.3098`.

## Next decision

A separately preregistered V62r1 measurement repair may inspect terminal-state handling in the residual checker while keeping the original V62 result, external models, task cells, policies, values, official rollouts, seeds, gates, and all other metrics immutable. No additional external rollout is authorized.

## Boundary

The failed qualification establishes no external-transfer claim by itself. It does not test SMC2 portability, generic POMDP scalability, continuous control, formal safety, human-authored language, or model/adapter performance. V58 remains deferred.
