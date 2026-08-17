# V73 non-harvestable calibration source-feasibility plan

## Purpose

V72 failed because its known-good calibration rock was also a rewarding control target. V73 must remove that shortcut before any planner is evaluated. The fresh development source is the Apache-2.0 IMPRL maintenance environment at commit `3c9cde75b48a2cba54f62330ead1e1dbc054d0cf`. Its source declares three finite damage states, component actions `do_nothing`, `replace`, and `inspect`, inspection-dependent observations, stochastic deterioration, replacement dynamics, and state-dependent system failure cost.

The selected slice is component 4 (zero-based index 3) of `hard-4-of-4_infinite.yaml`. Its transition matrix, initial belief, observation accuracy, replacement accuracy, action costs, failure factor, and discount are frozen directly from that configuration. The single-component failure cost is the component replacement cost multiplied by the declared failure factor. This projection is fixed before any policy value or action is computed.

## Project-authored adapter

V73 adds one observation-only `calibrate_beacon` action. It emits the selected source component's inspection distribution for a known healthy reference, costs the same inspection plus mobilisation charge, and applies the target's ordinary `do_nothing` deterioration row. The beacon cannot be replaced, harvested, or rewarded. A binary latent codebook either preserves source labels 0/1 or swaps them; failure label 2 and the collapsed null observation remain fixed. This is a source-grounded development adapter, not an unchanged external environment.

The no-inspection observation is collapsed to a single `none` symbol because the source distribution is state-independent in this environment. That is an exact belief-equivalent reduction, but it is still part of the project-authored exporter and therefore remains inside the claim boundary.

## Firewall and decision rule

Repository landing pages for IMPRL, jax-imprl, and optimality-of-decentralization were previewed during lead discovery. They are development-exposed. No simulator, policy, value, regret, information gain, protected source, human record, model, or adapter-training outcome may be accessed at this stage.

The source audit must bind the commit, Apache license, source code, configuration, selected component parameters, and V72 authorization. A pass authorizes only implementation of the frozen adapter and its preregistered structural dominance audit. It does not authorize exact Bayes-adaptive, MAP, posterior-sampling, open-loop, or myopic outcome evaluation.

References: [IMPRL](https://github.com/prateekbhustali/IMPRL), [maintenance environment source](https://github.com/prateekbhustali/IMPRL/blob/main/imprl/envs/structural_envs/k_out_of_n_infinite.py).
