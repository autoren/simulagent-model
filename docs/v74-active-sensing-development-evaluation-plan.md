# V74 one-shot source-grounded development evaluation plan

## Estimand

The single development model asks whether retaining posterior uncertainty over a persistent observation-label codebook has decision value when a non-harvestable calibration action is economically worthwhile. The exact Bayes-adaptive policy is compared with MAP certainty equivalence, persistent posterior sampling, best open loop, and a myopic policy under the exact joint posterior predictive.

## Preregistered policy structure

At horizon three, the exact policy must uniquely calibrate first, listen to the target after either beacon label, and then use both opening actions: `open_right` when the beacon and target labels match, and `open_left` when they differ. The cheaper beacon breaks the otherwise order-symmetric calibrate/listen tie; the exact raw root margin must be at least `0.02`.

MAP and both persistent posterior-sampling point models must listen to the target first because they treat the codebook as known. Their policies are evaluated without fallback under the true mixture. Each must incur at least `0.1` normalized regret. Exact planning must be strictly positive and beat best open loop by at least `0.015` normalized return; myopic control must also lose at least `0.015`.

## One-shot rule and boundary

The evaluator is locked before the durable attempt file is created. Exactly one result may be produced. Any failed scientific or integrity gate freezes a negative result and stops before protected-source discovery. A pass authorizes only a separately preregistered confirmation-design phase; it does not authorize selecting or scoring a holdout.

The result remains source-grounded development evidence. The high-fidelity noise setting, beacon, beacon cost, latent codebook, and observation collapse are project-authored, so this is not an unchanged external Tiger benchmark.
