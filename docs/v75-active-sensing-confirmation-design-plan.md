# V75 adapter and one-shot replication design

## Frozen adapter

The adapter preserves all four NOVA paint states and source transitions, rewards, discount, and target-inspection probabilities. It adds a zero-reward, identity-transition reference inspection of a known nonblemished condition and a binary canonical/reversed label codebook. The two uninformative labels emitted after source control actions are collapsed to one `none` symbol. This is a belief-equivalent representation change, not an added measurement.

Ten structural tests must establish source-array parity, normalization, common point-model support, exact sensor probabilities, reward semantics, non-harvestable calibration, fixed-policy value, open-loop value, and the registered resource envelope. Structural work may evaluate only the fixed lower-bound policy and all observation-independent sequences. It may not run the optimal joint planner, MAP, posterior sampling, myopic control, EIG, or any prior paint policy outcome.

## One-shot outcome gates

Only after a passing structural lock may the exact evaluator be implemented and frozen. The four-step joint-posterior policy is expected to treat reference-first and target-first sensing as exactly tied; deterministic tie-breaking selects `calibrate_beacon`. After reference-first sensing, every reachable label takes `inspect_target` second. Matching labels must lead to `paint` and then `ship`; differing labels must lead to `reject`. Both third-step controls must be reachable.

MAP certainty equivalence and persistent posterior sampling are preregistered to start with `inspect_target`, because a known codebook makes reference calibration decision-irrelevant. Their true-mixture regret, exact-over-open-loop advantage, and myopic regret must each exceed `0.015` of the frozen return scale. The point models must remain on common support with no fallback. This threshold is intentionally much lower than V74's `0.1` regret gate because the external paint source has only unit-scale rewards and `0.75` sensing accuracy; it is fixed from the V75 mechanism-level materiality threshold before optimized policies exist.

The result is one-shot. Failure is a negative replication, not permission to change the beacon, horizon, source, gates, or tie rule. Even on success, the claim is outcome-untouched external-domain replication rather than discovery-clean confirmation, because V68 previously inspected a malformed variant of the classic paint model.
