# V74 source-grounded Tiger adapter structural plan

## Frozen adapter

The adapter preserves pomdp-py's two Tiger states, uniform initial belief, source listening/opening transitions, configured `0.99` observation accuracy, opening rewards, target-listen cost, and `0.95` discount. It adds the prospectively screened non-harvestable beacon and a persistent canonical/reversed observation-label codebook. The joint initial belief is uniform over both codebooks and both Tiger locations.

Opening actions use one `none` observation because the source assigns `0.5` to either label after opening, independently of successor state; collapsing those two labels is exactly belief-equivalent. Calibration and target listening each retain both labels with positive probability under both point models. Rewards for opening depend on the pre-transition Tiger state exactly as in the source.

## Structural audit

Ten tests must bind shapes, normalization, source transitions, source rewards, configured observations, common support, the non-harvestable beacon, initial belief, exact belief-equivalent open-observation collapse, and the fixed policy schema. The audit may evaluate only the preregistered fixed policy and all 64 open-loop sequences. It may not call any optimal contingent, MAP, posterior-sampling, myopic, EIG, protected, human, or model procedure.

The fixed policy must reproduce the prospective economic advantage: calibrate, listen, open right on matching labels and left on differing labels. It must beat the best open loop by at least `5.0` raw return, exceed `0.015` of the full three-step return scale, and retain at least `0.005` normalized margin. A pass authorizes only evaluator implementation and locking; a failure stops before planner outcomes.
