# V72 RockSample one-shot development evaluation

The sole development model is the frozen 2×2 `RockSample.jl` export. At horizon four, compare exact joint Bayes-adaptive planning with joint MAP certainty equivalence, persistent posterior sampling, best open loop, and a myopic contingent policy. Evaluate every policy under the exact joint environment and normalize regret by the finite-horizon source reward span.

The intended positive mechanism is `check_reference` at the root, `check_target` after either reference observation, movement to the target, and history-dependent use of both `sample` and `east` as final controls. MAP is preregistered to choose `check_target` at the root. Exact must exceed MAP and persistent posterior sampling by normalized regret `0.02`, exceed open loop by normalized value `0.01`, and have root margin at least `0.01`. Both point controls must remain on common support with zero fallback.

This is a one-model external development screen, not confirmation. If any gate fails, V72 stops without changing its sensor efficiency, codebook noise floor, rewards, horizon, initial belief, model, controls, or thresholds. Protected-source discovery and all protected outcomes remain forbidden unless every development gate passes.
