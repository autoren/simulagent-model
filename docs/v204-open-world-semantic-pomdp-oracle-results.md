# V204 open-world semantic POMDP oracle results

## Verdict

V204 is a valid, informative negative result. The exact open-world planner had material advantages over MAP certainty-equivalence, persistent posterior sampling, myopic control, and the best open-loop sequence, but the preregistered process failed the abstention and forced-commit gates. The branch is frozen without changing its horizon, rewards, channels, hypotheses, comparators, or thresholds.

The decisive issue was a finite-horizon terminal loophole: repair actions had zero immediate reward and their good or bad consequence appeared only on a later `settle` action. A policy could therefore begin a repair at the last decision step and end the horizon before settlement. This made unresolved commitment artificially costless at the boundary.

## Locked result

- Exact root action: `calibrate`, tied with `inspect`.
- Exact value: `3.05366665`.
- Exact action after a root `green` observation: `calibrate`, not `defer`.
- Reachable exact-policy defer histories: `0`.
- Reachable repair actions: both `repair_A` and `repair_B`.
- Forced-commit normalized regret: `0.0`.
- MAP normalized regret: `0.06593386`.
- Persistent posterior-sampling normalized regret: `0.053734201975`.
- Exact normalized advantage over best open loop: `0.02526833325`.
- Exact normalized advantage over immediate deferral: `0.02526833325`.

Structural checks passed: all three semantic hypotheses had common observation support, calibration mutual information was `0.5824902351` nats, known-codebook inspection total variation was `0.89`, repair had no immediate reward, settlement carried the state-dependent consequence, beliefs normalized exactly, and no fallback was used.

## Gates

The following preregistered scientific gates failed:

1. The exact policy did not defer after a root `green` observation.
2. No defer action was reachable under the exact policy.
3. Forced commitment had no material regret.

Every access gate passed. V204 read no language records or raw model responses, loaded and ran no model, accessed no protected population, called no API or service, performed no training or ontology registration, mutated no trusted state, and executed no external action.

## Interpretation

V204 does support a limited mechanism claim: retaining an explicit outside-semantics hypothesis can improve exact sequential decisions relative to collapsing uncertainty to one point model or sampling and committing to one model. It does **not** establish the intended open-world abstention mechanism, because the process did not make all commitments terminally accountable.

This is not evidence that deferral lacks value. It is evidence that delayed consequences must be represented with a terminally proper objective. A finite-horizon benchmark cannot allow an unresolved commitment to escape its downstream cost simply because the horizon ends.

## Authorized successor

V204 itself authorizes no external candidate, language, or model run. The scientifically appropriate successor is a separately preregistered model-free oracle that preserves the conceptual question while repairing the process definition structurally. Acceptable formulations include:

- an absorbing pending state whose terminal value equals the eventual settlement value;
- a mandatory settlement phase outside the controllable horizon; or
- a fixed episode structure in which sensing/control decisions are followed by an unavoidable outcome realization.

The successor must add a gate proving that every repair commitment is settled on every trajectory and that no policy can gain by postponing consequences beyond the horizon. Only after that model-free mechanism passes should the project open a fresh-source design track.

## Frozen decision

`freeze_V204_negative_without_parameter_horizon_reward_channel_or_gate_tuning`
