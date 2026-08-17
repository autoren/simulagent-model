# V75 outcome-untouched active-sensing replication result

## Result

V75 passed its prospective economic screen, source-preserving adapter audit, and sole exact replication attempt. The result reproduces the V74 mechanism in a materially different manufacturing-control system, with lower source-native sensing accuracy and multi-step paint/ship decisions. It is an outcome-untouched external-domain replication, not discovery-clean confirmation.

The source was the MIT-licensed NOVA `paint_95.pomdp` file at commit `fa6f0bf038509cb7bb94fb79e38e691c6e6d83e9`. It supplied four manufacturing states, `paint/inspect/ship/reject` dynamics, a `0.75/0.25` target-inspection channel, unit state-dependent rewards, reset transitions, and discount `0.95`. The project layer added a zero-reward, identity-transition reference inspection of a known nonblemished condition and a persistent canonical/reversed `NBL/BL` codebook. Deterministic, uninformative source control observations were collapsed to `none` without changing belief updates.

## Prospective and structural gates

The fixed reference-inspect-contingent-control lower bound had value `0.1663984375`, versus `0` for the best open loop. Its normalized advantage was `0.0224264`, leaving `0.00742642` margin over the registered `0.015` threshold. These quantities were frozen before an adapter or optimized planner existed.

All ten structural tests passed. Source transition and reward arrays were preserved exactly; both latent observation models had identical support; calibration mutual information was `0.130812` nats; target-inspection total variation was `0.5`; paired decision accuracy was `0.625`; the dense kernel occupied 2,240 bytes; and the four-step Bellman upper bound was 400 nodes. The adapter exactly reproduced the economic lower bound and the best of all 625 open-loop sequences was four zero-reward beacon actions.

## Exact replication outcome

The exact joint-posterior policy had value `0.1663984375`. Reference-first and target-first sensing were exactly tied at the root; deterministic tie-breaking selected `calibrate_beacon`. After either reference label, the policy inspected the target. Matching labels led to `paint` and then `ship`; differing labels led to `reject`. Both state-dependent controls were reachable. The root margin to the best nonoptimal action was `0.0592266` raw, or `0.00798229` normalized.

MAP certainty equivalence and persistent posterior sampling both inspected the target immediately. Evaluated under the true joint mixture, each had value `-0.004286875` and normalized regret `0.0230042`. Both remained on common support and no fallback was used. Myopic control and best open loop each had value `0`, giving exact planning normalized advantage/regret `0.0224264`. Every registered scientific and integrity gate passed on attempt one of one.

## Interpretation and limits

V75 strengthens the mechanism claim beyond the V74 Tiger development model. The effect survives when sensing accuracy falls from `0.99` to the source's `0.75`, when correct action requires a paint-then-ship sequence, when rewards are unit-scale rather than dominated by a `-100` catastrophe, and when reference-first is tied rather than uniquely optimal. The useful invariant is not a particular first action: it is maintaining the joint codebook/state posterior until observations can be compared and control can branch.

The evidence remains bounded. V75 is not an unchanged external benchmark because the reference beacon and latent codebook are project-authored. It is also not source-discovery clean: V68 previously audited and rejected a malformed five-state POBAX variant of the classic paint model before any policy outcome. V75 used a separately pinned, valid four-state MIT source and never accessed a prior paint policy value, regret, or EIG result. The proper label is therefore **outcome-untouched external-domain replication**, not independent confirmation.

The branch is closed to tuning and rerun. The right successor is synthesis plus a new discovery-clean source program. A broader confirmation should use a domain family not previously inspected by the project, retain common support, supply its own sensing and delayed control economics, and pass a source-level lower bound before any adapter or policy optimization.

References: [NOVA](https://github.com/kylewray/nova), [pinned paint source](https://github.com/kylewray/nova/blob/fa6f0bf038509cb7bb94fb79e38e691c6e6d83e9/tests/benchmarks/algorithms/domains/paint_95.pomdp).
