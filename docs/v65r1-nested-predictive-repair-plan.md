# V65r1 preregistration: pre-subset nested-predictive bias repair

Date frozen: 2026-08-16

## Why a repair is required

The committed V65 design requires each outer `(identity, theta)` particle to use its own 127-particle
inner state posterior directly when estimating the next-observation distribution. During candidate
implementation testing—before an implementation lock, before materializing any V65 subset, and
before any candidate evaluation—a synthetic public-history fixture exposed a structural nested
Monte Carlo bias.

Even when the true conditional maze-state distribution is independent of theta, independently
randomized finite inner filters produce slightly different state distributions for different theta
particles. Plug-in mutual information treats those differences as signal about theta. Increasing the
number of outer particles does not remove the bias because every static particle carries a fresh
finite-inner-filter error. On the root fixture, the frozen exact EIG vector is approximately
`[0.02798, 0.00437, 0.02798, 0.00433]`, while a 509-particle, three-repeat plug-in computation is
approximately `[0.03977, 0.01244, 0.04069, 0.00819]`.

This is a design defect, not a V65 result. No V64 record has been selected or evaluated under V65.
The original V65 design lock remains immutable and is explicitly superseded only at the acquisition
conditional-state interface described here.

## Sole repair

V65r1 retains the three-repeat equal-weight pooled SMC² posterior weights over identity and theta.
It also retains the 127-particle inner filters inside SMC² for likelihood estimation, identity
evidence, outer weighting, resampling, and PMMH. The pooled particle-state marginal remains a primary
posterior-accuracy target under every original V65 gate.

For acquisition only, V65r1 Rao–Blackwellizes the dynamic nuisance state. Conditional on each pooled
SMC² `(identity, theta)` atom and the public reset/action/observation history, it runs the known exact
11-state forward filter. It then propagates that conditional distribution through the atom's
candidate transition and the pinned observation kernel. The weighted collection of these
conditional predictives defines mutual information about `(identity, theta)`.

Thus uncertainty about the static dynamics remains approximate and particle-based; only the small,
known finite-state conditional is integrated exactly. Particle ancestry, inner state, and repeat
identity remain forbidden information targets. The original plug-in inner-particle predictive is
retained as a mandatory negative control and diagnostic.

## Independent pre-subset feasibility gate

The repair auditor independently constructs the public root history with reset observation `left`.
Across 12 fresh fixture replications it creates 509 outer particles per identity, three repeats, and
127 independently sampled inner states per outer particle without importing the candidate V65
implementation. It compares:

1. exact V64 EIG;
2. EIG using the noisy plug-in conditional state per static particle; and
3. EIG using the exact conditional 11-state filter per static particle while preserving the same
   approximate static weights.

The repair is eligible only if mean plug-in EIG-vector bias is at least `0.005` nats and mean
Rao–Blackwellized error is at most `0.001` nats. This fixture is not an evaluation population and
uses no V64 selection record, audit field, result, or V65 subset seed.

## Everything else remains frozen

V65r1 makes no change to the 48-record hash-selection rule, particle budgets `[31, 127, 509]`,
127-particle inner filters, three-repeat pooling, resampling, PMMH, random roots, exact reference,
posterior or predictive targets, EIG and regret thresholds, scaling gates, controls, access counters,
one-shot hierarchy, or downstream authorizations.

A later pass must be described as **Rao–Blackwellized SMC² EIG portability**. It will not establish
that a pure nested-particle plug-in predictive is calibrated, nor will it authorize sequential
approximate adaptive rollouts, reward planning, formal verification, human data, model access, or
training except through the original V65 staged decision hierarchy.
