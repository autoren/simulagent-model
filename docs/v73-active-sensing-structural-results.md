# V73 source-grounded maintenance structural results

## Bottom line

V73 is a useful structural negative result. It fixed V72's harvestable-calibration confound, but the preregistered adaptive advantage was too small to authorize optimal planner evaluation. The branch stops before exact Bayes-adaptive, MAP, posterior-sampling, myopic, or protected-source outcomes.

## What passed

The pinned [IMPRL](https://github.com/prateekbhustali/IMPRL) component slice exported cleanly to a three-state, four-action, four-observation exact kernel. All ten structural tests passed. The calibration beacon applies the source `do_nothing` transition, costs `-8`, cannot be harvested or controlled, and has no positive reward. Across the complete reward tensor, the maximum immediate reward is zero and the number of strictly positive entries is zero.

The intended information structure is real:

- calibration mutual information is `0.3680642071684971` nats;
- healthy-versus-degraded inspection total variation is `0.8500000000000001`;
- the two latent codebooks have identical observation support;
- paired same-label and different-label histories imply degraded-state proxy posteriors `0.1276595744680851` and `0.7522935779816514`;
- those values straddle the frozen replacement threshold `0.34814814814814815`.

Thus the beacon can resolve label meaning, the target sensor can distinguish maintenance-relevant states, and the evidence can cross a later control threshold. V73 did not fail for the V71 dominant-action reason or the V72 harvestable-reference reason.

## What failed

The fixed five-action adaptive policy had exact value `-118.941844504576`. The best of all 1,024 open-loop sequences was:

```text
do_nothing → do_nothing → replace_target → do_nothing → do_nothing
```

with value `-119.43454720000003`. The fixed adaptive policy therefore improved raw value by only `0.492702695424`, or `0.0004026591008991246` of the frozen return scale. The preregistered minimum was `0.005`.

This single failed noncompensatory gate closes V73. No optimal contingent planner was called, so V73 supplies no BA-versus-MAP or posterior-sampling comparison.

## Interpretation

Removing an obvious reward shortcut was necessary but not sufficient. Under the selected source costs, discount, deterioration, and five-action horizon, a predetermined replacement schedule captures nearly all available value. Calibration plus inspection changes the appropriate maintenance decision, but the expected gain is too small relative to the full consequence scale to justify a downstream study.

The correct successor must screen the *economic strength* of information before implementation, not just its mutual information or threshold-crossing ability. A fresh source should satisfy a source-level lower bound in which the expected reduction in maintenance decision loss, after paying all sensing and delay costs, clears the material-effect threshold by margin. The V73 repository, component, parameters, adapter, horizon, fixed policy, and gates are closed and may not be tuned or reused for successor outcomes.

V73 remains source-grounded development work because the non-harvestable beacon, codebook uncertainty, null-observation reduction, and single-component failure projection are project-authored. It is not unchanged external-environment evidence, confirmation evidence, approximate-inference evidence, or human evidence.
