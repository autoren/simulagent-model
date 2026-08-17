# V74 source-grounded active-sensing development result

## Result

V74 passed its prospective economic screen, adapter structural audit, and single immutable exact development evaluation. It is the first sensor-codebook branch in this sequence where the source-grounded sensing economics were required to pass before an adapter or optimal planner existed.

The fresh development source was pomdp-py's MIT-licensed Tiger problem at commit `bd0e4392247aebfe9a95b449275237dcc25e7737`. Source structure supplied the two hidden states, listen/open actions, symmetric dynamics, parameterized noisy observations, `+10/-100` state-dependent opening rewards, `-1` target-listen cost, and `0.95` discount. V74 fixed the source-exposed observation noise at `0.01`. The non-harvestable reference beacon, its `-0.5` cost, and the latent canonical/reversed label codebook were project-authored.

## Economic and structural gates

Before adapter code, the fixed calibrate-listen-open policy had prospective value `5.609355`, versus `-1.42625` for the best open loop. Its raw advantage was `7.035605`, or `0.0224225` of the frozen three-step return scale, leaving `0.0074225` margin over the registered `0.015` threshold. The paired codebook-and-state decision was correct with probability `0.9802`, above the `0.9091` break-even probability for opening a door.

The implementation then reproduced those numbers exactly. All ten structural tests passed. The dense kernel occupied 640 bytes; the horizon-three Bellman upper bound was 43 nodes; the two point models had identical support; calibration mutual information was `0.637146` nats; target-listen total variation was `0.98`; and the best of all 64 open-loop sequences was three beacon actions. The beacon had no positive or harvestable reward.

## Exact development outcome

The exact joint-posterior policy uniquely chose `calibrate_beacon` at the root, with value `5.609355`. Its raw margin over `listen_target` was `0.0250001`. After either beacon label it chose `listen_target`; matching beacon/target labels led to `open_right`, differing labels to `open_left`. Both final controls were reachable.

MAP certainty equivalence and both persistent posterior-sampling point policies instead chose `listen_target` immediately. Evaluated under the true joint mixture, each had value `-44.20125` and normalized regret `0.158746`. Both remained on-support and no fallback was used. Best open loop and myopic control each had value `-1.42625`; exact planning exceeded each by `0.0224225` normalized return. Every preregistered scientific and integrity gate passed on the sole attempt.

## Interpretation and limits

V71 showed that label uncertainty can be irrelevant when one action dominates. V72 showed that a rewarding reference can bypass calibration. V73 showed that non-harvestability and threshold crossing remain insufficient when information is not economically valuable. V74 closes that design loop in a development model: when calibration is non-harvestable, sufficiently accurate, cheaper than target listening, and necessary to avoid a large state-dependent loss, preserving codebook uncertainty changes the contingent policy and produces material value.

The result is intentionally narrow. It is not an unchanged external Tiger benchmark, not confirmation evidence, and not support for the particular `0.99` accuracy or `-0.5` beacon cost outside this configured model. Those choices were prospectively screened and frozen, but they are still project-authored. The small root margin also shows that calibration-first ordering depends on the registered cost asymmetry; the large policy-value separation comes from the later MAP commitment error, not from a large first-action Q margin.

The correct successor is a fresh confirmation-design phase, not another V74 tuning run. It must preregister source discovery and seek an independently supplied environment or configuration with source-native high-fidelity sensing, a non-harvestable reference or calibration action, delayed state-dependent control loss, and an economic lower bound that passes before exact outcomes. V74's source commit, noise, beacon cost, horizon, adapter, controls, and gates are now closed.
