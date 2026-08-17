# V73 source-grounded maintenance development plan

## Hypothesis

If the frozen structural audit passes, exact Bayes-adaptive planning should calibrate the non-harvestable beacon, inspect the target, and condition maintenance on the observation history. MAP certainty equivalence should treat its selected label codebook as known and inspect the target immediately. Persistent posterior sampling may preserve uncertainty across episodes, but it cannot combine evidence across the two codebooks within one policy and is therefore expected to lose value.

## One authorized outcome

The already frozen horizon is five actions. Exact Bayes-adaptive, MAP, persistent posterior sampling, best open loop, and myopic control will be evaluated once under the exact joint prior and exact source-grounded kernel. The root action and margin, calibration branches, reachable maintenance actions, exact values, normalized regrets, support, fallback count, and Bellman-node accounting will be recorded.

All gates are noncompensatory. Exact must choose `calibrate_beacon`; MAP must choose `inspect_target`; both calibration observations must lead to target inspection; reachable histories must use both `do_nothing` and `replace_target`; MAP regret, posterior-sampling regret, and adaptive-over-open-loop advantage must each reach `0.005` of the frozen return scale. Point-model control must remain on common support with zero fallback.

A failure is frozen and closes V73. A pass would authorize only fresh protected-source discovery. It would not turn this project-authored calibration/codebook layer into an unchanged external benchmark, and it would not authorize approximate inference, SMC², human substitution, model access, or adapter training.
