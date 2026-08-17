# V68 development-only exact sensitivity screening

## Purpose

Before any untouched POBAX model is scored, this stage asks whether the unchanged V64 command-
channel uncertainty family produces reward decisions for which exact Bayes-adaptive planning is
materially different from collapsing to a MAP model. It uses only four models already exposed in
V62–V67: nonterminating 4x3, tiger-alt-start, T-maze 2, and T-maze 5. This is a design feasibility
screen, not a confirmatory replication.

The static latent remains a balanced forward-versus-backward adjacent command substitution and a
scaled Beta(2,2) fidelity parameter on `[0.60, 0.95]`. The supplied environment transition,
observation, reward, initial distribution, and discount are unchanged. Per-model canonical action
cycles are fixed in the config before execution. The family is best described generically as command-
channel uncertainty; only cardinal-action environments admit the earlier actuator interpretation.

## Complete development census

Each model contributes its empty public history and every reachable one-action/one-observation
history obtained by crossing all frozen actions with every strictly positive observation. No history
is deduplicated, selected, rejected, or replaced. There is no project-authored reset observation;
the root joint belief is the static prior times the pinned source start distribution.

At each retained belief, the primary 65-node exact quadrature computes a horizon-three Bayes-
adaptive reward policy. A 129-node run checks convergence. Controls are MAP collapse, deterministic
17-point posterior-sampling quadrature with one model persistent for the complete policy, the best
length-three open-loop action sequence, myopic reward, and myopic information-only selection.

## Frozen materiality rule

Because source rewards use different units, regret is divided by
`max(1, (r_max-r_min) * (1 + discount + discount^2))`. A MAP regret of 0.005 on this scale is
material. Advancement requires at least three root-action disagreements, at least two materially
wrong MAP records, maximum normalized MAP regret of at least 0.01, at least two material open-loop
regrets, and at least one material posterior-sampling regret. Primary and convergence quadrature
values must agree to normalized error `1e-8`, and every primary action must belong to the
convergence optimal set.

Failure stops the unchanged-family route before any holdout is scored. It may motivate a new
prospectively registered family revision, but it cannot justify selecting favorable holdout models
or changing these gates after the fact. A pass authorizes design of the untouched-model evaluation;
it is not itself evidence of multi-environment replication or approximate-inference portability.
