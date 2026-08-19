# V205 terminally proper open-world semantic POMDP results

## Verdict

V205 is a positive model-free mechanism result. Every preregistered scientific, terminal-accounting, integrity, and access gate passed.

The exact open-world policy calibrates first, inspects the target after `red` or `blue` calibration evidence, and safely defers after `green`, where the posterior is dominated by the explicit `OUTSIDE_UNKNOWN` semantic hypothesis. On later informative histories it reaches both `repair_A` and `repair_B`. Every repair receives an unavoidable automatic state-dependent settlement; no repair or sensing path can escape its consequence at the episode boundary.

## Exact policy

- Root action: uniquely `calibrate`.
- Root value: `2.133240000000001`.
- Root Q-values: `calibrate = 2.13324`, `defer = -2.0`, `inspect = -2.5`, and either immediate repair `= -10.0`.
- After root `red`: `inspect`.
- After root `blue`: `inspect`.
- After root `green`: `defer`.
- Reachable selected actions: calibration, inspection, deferral, and both repairs.
- Reachable defer histories: `3`.
- Exact terminal paths: `7`, comprising `4` automatic settlements and `3` safe deferrals, with `0` unsettled paths.

## Comparator results

Using the frozen return scale of `120`:

- Exact advantage over immediate deferral: `0.0344436667` normalized.
- Closed-world regret: `0.0149436667`; it incorrectly chooses `inspect` after root `green`.
- Forced-commit regret: `0.0149436667`.
- MAP certainty-equivalence regret: `0.105277`; its root action is `inspect` and its true-mixture value is `-10.5`.
- Persistent posterior-sampling regret: `0.084027`; its true-mixture value is `-7.95`.
- Exact advantage over the best open-loop program: `0.0344436667`; the best open-loop program is immediate deferral.
- Immediate-reward myopic regret: `0.1011103333`; it repairs immediately and receives true-mixture value `-10.0`.

The closed-world and forced-commit policies have the same value in this construction because both fail specifically on the branch where outside-semantics evidence makes deferral optimal.

## Structural evidence

- Calibration mutual information: `0.5192531842` nats.
- Known-codebook inspection total variation: `0.90`.
- Minimum sensing-support probability: `0.02`.
- Belief normalization rate: `1.0`.
- Mandatory automatic settlement rate: `1.0`.
- Unfinished-sensing safe-deferral rate: `1.0`.
- Horizon escape paths: `0`.
- Unsettled repair terminals: `0`.
- Fallback count: `0`.

## What V205 establishes

V205 demonstrates, in a deliberately small exact model, that explicit open-world semantic uncertainty can change both information gathering and control:

1. a reference observation first identifies whether the sensor semantics are usable;
2. usable semantic evidence makes target inspection worth its cost;
3. evidence for an outside or unmodeled semantic regime makes further inspection valueless and safe deferral optimal; and
4. collapsing or forbidding that outside option creates measurable loss.

This is stronger than V71's sensor-semantic negative result because the V205 uncertainty is control-relevant, and stronger than V204 because all commitments are terminally accountable.

## Boundaries and next authorization

The priors and likelihoods are project-authored oracle quantities. They are not learned from LLM ranks, model confidence, human responses, or external language. V205 therefore does not establish that an LLM supplies calibrated likelihoods, detects open-world semantics, or knows when to abstain.

No language records or raw model responses were read. No model was loaded or generated from; no protected data, API, training, ontology registration, trusted-state mutation, service call, side effect, action, or execution occurred.

The positive result authorizes only a separately preregistered metadata/source feasibility design for an externally grounded analogue. It does not authorize immediate language extraction, candidate evaluation, or an LLM run.

## Frozen decision

`freeze_V205_positive_terminally_proper_open_world_abstention_mechanism_and_authorize_separate_fresh_source_feasibility_design_only`
