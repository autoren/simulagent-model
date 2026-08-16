# V62r1 preregistration: terminal Bellman-residual measurement repair

## Immutable source result

V62 remains a sealed 31-of-32-gate failure. Its candidate and independent scalar planner agreed on every registered root decision, the unchanged POBAX runtime completed all 24 rollout cells within the simultaneous familywise 99% bounds, and every control separation passed. The only failed gate was the independent Bellman residual, which was `1.0` for Tiger at horizons 5 and 7.

A labeled post-hoc diagnostic reproduced every V62 cell exactly and localized all four residual failures among 66 reachable belief nodes. Every failing belief had support entirely on all-action absorbing states. There were no nonterminal residual failures. The planner stored zero value and zero action values at those terminal nodes. POBAX marks an all-action absorbing successor done, and the frozen rollout collects no later reward from a done episode.

## Sole permitted repair

V62r1 adds the missing terminal base case to a new, independent residual checker. At horizon zero or when all positive-probability states are absorbing under every action, the checker verifies that the stored decision value and every stored action value are zero. At every other belief it independently recomposes each action value from scalar transition, reward, observation, normalization, discount, and child-value operations. The implementation may not call V62's old residual, expected-reward, observation-distribution, or belief-update helpers.

Five analytic fixtures and six targeted mutants must pass before the rescore is authorized. The mutants cover omission or weakening of terminal detection, failure to inspect terminal values or action values, and incorrect discount or observation semantics on nonterminal backups.

## One immutable rescore

The single repair rescore visits the same six task cells and 66 reachable nodes. It must reproduce the old per-cell residuals exactly, show that every old failure is terminal-only, reduce the corrected maximum residual to at most `1e-10`, reproduce all other 31 V62 gate checks exactly, and byte-bind the original exact records, official rollout records, result, models, locks, and attempt count. No candidate evaluation or external rollout is repeated.

If all 12 noncompensatory repair gates pass, the combined V62/V62r1 evidence supports the narrow external exact finite-state, finite-horizon belief/planning transfer claim on the three pinned POBAX instances. V62 itself remains failed and V62r1 is a measurement repair over the same artifacts, not an independent replication. Any nonterminal old failure, record drift, rollout drift, broader metric change, fixture failure, or surviving mutant rejects the repair.

This does not establish SMC2 portability, unknown-program inference, general POMDP scalability, continuous or long-horizon control, formal safety, human-language robustness, or model/adapter performance. V58 remains deferred.
