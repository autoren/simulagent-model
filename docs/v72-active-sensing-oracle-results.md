# V72 shared-support active-sensing oracle results

## Bottom line

The preregistered engineered mechanism check passed all 13 oracle gates in its single authorized run. This verifies that the locked exact-planning implementation can express a fallback-free sensor-codebook problem where preserving uncertainty changes the best action. It is an implementation oracle, not external, development, or confirmation evidence.

## Positive mechanism fixture

- Exact Bayes-adaptive root action: `calibrate`.
- Exact value: `2.6000000000000014`; root action margin: `5.600000000000001`.
- After either calibration observation, the exact policy chose `inspect`; across the next observations it used both `repair_A` and `repair_B`.
- MAP and persistent posterior sampling both chose `inspect` at the root and had exact-environment value `-6.0`.
- Exact minus MAP and posterior-sampling regret was `8.6`, or `0.09555555555555557` of the locked finite-horizon return scale.
- Best open-loop and myopic exact-environment values were both `-3.0`.

The two codebooks had identical full observation support. Calibration and inspection each carried `0.36806420716849714` nats in the relevant binary channel, all point-model branches stayed on-support, and fallback count was zero.

The seven-state Markov representation makes a repeated `calibrate` action in the already-calibrated phase repeat the reference reading. This behavior was declared in the implementation diagnostics. At the locked three-action horizon it was legal but noncompetitive: every exact branch inspected second.

## Dominant-action negative control

The control chose `repair_A` under exact Bayes-adaptive, MAP, posterior sampling, and myopic control. Exact, MAP, and posterior-sampling values were all `5.0`; both normalized regrets were exactly `0.0`. Thus shared-support uncertainty alone did not manufacture an adaptive advantage when one control action dominated.

## Claim boundary and next authorization

This result establishes mechanism availability only. The oracle used two engineered fixtures, no human records, no model or adapter calls, no SMC2, no V71 protected access, and no external candidate metadata or outcomes.

The next authorized step is metadata-only discovery of fresh external active-sensing sources. Candidate policy values, optimal actions, regrets, and expected information gain remain forbidden until an external source inventory, structural feasibility gate, partition, and evaluator are separately frozen.
