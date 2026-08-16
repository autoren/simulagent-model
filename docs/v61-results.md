# V61 results: bounded long-horizon approximate-belief policy verification

**Qualification:** PASS  
**Frozen policies:** 72 (24 each at horizons 3, 5, and 7)  
**External checker:** Storm 1.13.0  
**Claim boundary:** bounded exact-posterior execution of the frozen V60 policies; not search optimality, worst-case safety, an unbounded guarantee, or language robustness

## Main findings

All 72 primary V60 policies reconstructed exactly: tree-hash, root-action, and search-metadata match rates were each `1`. Their explicit models contained `1,722,301` states and `2,327,207` transitions in total.

Storm completed every model. Its maximum termination-probability error was `0`, maximum success-probability error against the independent executor was `4.64287497337e-11`, and maximum expected-return error was `4.90034679501e-11`.

The independent reachable-state checks covered `2,229,218` invariants and `1,215,096` transition supports. Invariant, exact support/probability, and public-history policy-totality rates were all `1`, with zero deadlocks and zero Z3 unknown results.

Every stored V60 2,048-episode policy mean fell inside its preregistered familywise 99% Hoeffding radius. The largest exact-vs-Monte-Carlo error was `0.0212783203125` and excess over the simultaneous bound was `0`.

## Exact verified return by horizon

| Horizon | Mean | Minimum | Maximum |
|---:|---:|---:|---:|
| 3 | 0.30050326534 | 0.00457863041458 | 0.48 |
| 5 | 0.515427888534 | -0.01 | 0.99 |
| 7 | 0.419446949913 | -0.0525258460618 | 0.97 |


## Interpretation

V60's approximate-belief policies are no longer supported only by sampled execution estimates. Across the exhaustive frozen census, a separately implemented exact executor and an external probabilistic model checker agree on their complete bounded execution semantics through horizon seven. This verifies the deployed policies in the frozen symbolic domain. It does not prove that UCT found an optimal policy, define or verify catastrophe avoidance, make a guarantee uniform over all continuous parameter values, or replace the deferred human-authored language track.

All 27 noncompensatory gates passed. Bundle hashes, source-result hashes, tool versions, attempt count, and the zero-truth-access firewall were independently recomputed after the run.
