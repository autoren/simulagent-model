# V72 external RockSample development results

## Bottom line

V72 is a valid and informative negative development result. The engineered oracle proved that the planner can express the intended shared-support active-sensing mechanism, but the prospectively selected external RockSample blueprint did not instantiate it. Seven of nine noncompensatory development gates failed, so V72 stops before any protected-source discovery or confirmation outcome.

## What happened

Exact Bayes-adaptive, MAP certainty equivalence, persistent posterior sampling, and best open loop all selected the same root action: `west`. The best open-loop sequence was `west → sample → east → east`, with exact value `13.786874999999998`. The independent audit reproduced this directly as `0.95 × 10 + 0.95³ × 5 = 13.786874999999998`.

The sequence moves to the known-good reference rock, samples its guaranteed `+10` reward, and then exits for `+5`. Because the calibration reference was itself a harvestable source of reward, uncertainty about sensor labels was irrelevant to the best plan.

- MAP normalized regret: `0.0`.
- Posterior-sampling normalized regret: `0.0`.
- Exact-over-open-loop normalized advantage: `0.0`.
- Exact root margin: `4.761874999999998`.
- Calibration-channel mutual information was nonzero (`0.08228287850505195` nats), but it had no decision value.

Both point models retained identical observation support, their on-support rate was `1.0`, and fallback count was zero. Thus this is another control-relevance boundary, not a fallback artifact or implementation failure.

## Gate result

Only common-support/zero-fallback and the root-margin gate passed. The required exact and MAP root actions, reference-then-target branch structure, distinct final controls, MAP regret, posterior-sampling regret, and adaptive-over-open-loop advantage all failed.

## Correct successor constraint

A successor must prevent the calibration reference from being a rewarding control target. Suitable designs include a known-bad reference, a non-harvestable calibration beacon, or an observation-only reference state. Before any policy outcome, a structural dominance audit must enumerate immediately harvestable known rewards and reject a design if an open-loop route can bypass the sensing decision. Sensor discriminability and the final good/bad control threshold must also be certified from source parameters before evaluator locking.

No V72 parameter, reward, horizon, model, control, or gate may be changed retrospectively. No protected confirmation source was selected or scored; no EIG, SMC2, human, model, or adapter work occurred.
