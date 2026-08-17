# V72 external active-sensing source-feasibility plan

## Purpose

Find fresh external environments whose declared source structure makes sensor semantics control-relevant before any planner outcome is computed. This stage may pin repositories, licenses, model paths, dimensions, action roles, observation dependencies, reward roles, and exact-export availability. It may not run a candidate simulator or compute a candidate policy value, optimal action, regret, or expected information gain.

## Prior-exposure disclosure

During initial planning, the public landing pages for `sisl/SBO_AIPPMS`, `AdaCompNUS/sarsop`, and `sisl/BetaZero.jl` were previewed as general leads. No model file or policy/value output was inspected by the oracle evaluator. These repositories are nevertheless treated as exposed and development-only. They cannot provide protected confirmation evidence.

## Structural admission rule

A candidate is eligible only if source metadata establishes every noncompensatory property below:

1. A pinned source and usable license.
2. Model files not used in V62–V71.
3. Distinct sensing and control actions.
4. An observation channel that depends on sensing action and hidden state.
5. A delayed, state-dependent consequence of control.
6. At least two hidden states for which different control actions are semantically appropriate.
7. A finite exact representation or deterministic finite exporter compatible with an independently auditable exact planner.

Eligibility must be decided without candidate policy values, best actions, regret, or EIG. Repository popularity, benchmark results, and reported algorithm performance are not selection variables.

## Partition and next gate

All sources inspected now are development-exposed. A later confirmation partition must use a fresh repository or prospectively sealed untouched model files that are neither named nor parsed during development. After this inventory is frozen, the next stage may define a deterministic transformation and resource census, but it still may not compute candidate outcomes until a separate evaluator lock exists.

If the configurable `RockSample.jl` source passes the inventory gate, the development blueprint is fixed to a 2×2 map with a known-good reference rock at `(1,1)`, an uncertain target at `(2,2)`, and the robot initially at `(2,1)`. A binary latent codebook swaps only `good`/`bad` check labels. A fixed `0.2` uniform noise-floor mixture keeps those labels on common support even when the source sensor is colocated with a rock; the source `none` observation for movement, sampling, and exit remains unchanged. Source rewards are fixed prospectively to `+10/-10` for good/bad sampling, `+5` for exit, and `-0.5` per check over a four-action horizon. The intended structural path is reference check, target check, move north, then sample-or-exit. That sequence is a design description, not a predicted optimal policy.
