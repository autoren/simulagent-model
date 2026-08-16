# V55 preregistration: short-horizon exact Bayes-adaptive planning

V54 passed all 25 sealed exact-EIG, evidence-efficiency, adaptive-calibration, control, and selection-integrity gates. Its outcome lock authorizes only preregistration of short-horizon exact Bayes-adaptive planning. V55 uses that authorization without adding approximation, learned search, formal verification, language, or model access.

## Decision problem

V55 starts from the validated joint belief over program identity `M`, continuous stochastic parameter `theta`, and the hidden current world/queue configuration. Each task declares a terminal goal: one world atom must equal a specified Boolean after three actions. Success pays `1`, failure pays `0`; `pulse` and `route` each cost `0.01`, while `wait` is free.

The agent has two entities and therefore five legal actions at every decision: `wait` plus `pulse` and `route` for both ordered distinct actor/target bindings. It observes the full world after every action, but scheduled-event queues remain hidden. The program and theta remain static latent variables. Three actions are long enough for the existing one- and two-tick delayed effects to matter while keeping exact belief-tree enumeration tractable.

For horizon `h`, the primary planner implements

\[
V_h(b)=\max_a\left[-c(a)+\sum_o p(o\mid b,a)V_{h-1}(\tau(b,a,o))\right],
\]

with `V_0(b)` equal to the belief probability that the terminal goal is satisfied. Future actions may depend on observed outcomes; the root action may not depend on truth fields or future observations.

## Exact references

The primary implementation is a memoized recursive belief-tree dynamic program over all eight programs, 257 theta quadrature nodes, hidden configurations, legal actions, and exact observation branches. A separately structured scalar policy-tree enumeration must reproduce every root value and optimal set. An independent exact policy evaluator executes the selected contingent policy from the root belief and must reproduce its Bellman value.

Analytic fixtures cover horizon zero, exhaustive horizon one, reduction to open-loop control under uninformative observations, dominance over registered deployable baselines, and delivery of a two-tick delayed effect within the three-action horizon.

## Baselines and scientific test

The registered deployable baselines are exact open-loop posterior-predictive planning, greedy reward, MAP-program planning, posterior-mean-theta planning, EIG-only selection, and a planner that disables within-horizon learning about static latents. A clairvoyant planner conditioned on the true latent state is an upper bound only.

Exact Bayes-adaptive value must never fall below any deployable baseline under the same root belief, nor exceed the clairvoyant bound. Beyond weak dominance, at least 15% of tasks must show an adaptive-over-open-loop value gap of `0.005`, and the mean gap must be at least `0.002`. The protocol also requires non-myopic root choices, information-then-control policies, and sensitivity to delayed consequences on at least 10% of tasks each. These are separate non-compensatory gates: exact arithmetic alone is not enough if the task population never makes adaptive planning useful.

## Population and firewalls

Population construction remains forbidden until the planner, scalar reference, independent policy evaluator, baselines, controls, and leakage guards pass altered-seed implementation audits and are frozen.

The future population contains 32 two-entity tasks. Each of the eight generating templates appears four times, but the balanced truth assignment is shuffled by a seed independent of history and goal generation. Sixteen tasks have prior-like all-wait histories and sixteen have mixed informative histories. Goal truth values are exactly balanced. Public histories must be fresh against V49–V54.

Controls remove contingency, truncate to greedy reward, collapse program uncertainty, collapse theta uncertainty, optimize EIG instead of reward, disable static-latent belief updates, or attempt to leak a future outcome. At least five of seven must be detected or dominated; leakage counts only if rejected.

A sealed pass authorizes only preregistration of symbolic and probabilistic verification of the frozen finite-horizon policy. It does not itself authorize formal verification, longer horizons, approximate or learned planning, language grounding, open ontologies, model access, training, or a final claim.
