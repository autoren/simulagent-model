# V204 open-world semantic POMDP oracle plan

## Question

Can an exact uncertainty-aware controller use an explicit outside-semantics hypothesis to decide among further
sensing, state-dependent control, and safe deferral when the consequence of control is delayed?

V74/V75 already established that retaining uncertainty over two label codebooks can change control. V204 adds the
missing open-world branch. `CANONICAL` and `REVERSED` codebooks encode conditions A/B with high fidelity. The
`OUTSIDE_UNKNOWN` hypothesis emits a green-dominant channel that carries no condition meaning. Every codebook assigns
positive probability to red, blue, and green, so no policy benefits from off-support fallback.

## Fixed process

The hidden condition is A or B. `calibrate` observes a known A reference; `inspect` observes the target. `repair_A`
and `repair_B` move to hidden pending-good or pending-bad states but give no immediate reward. Only the later `settle`
action reveals `+10` or `-30`. `defer` safely terminates for `-2`. Sensing costs `-0.5` and the horizon is five.

The full exact planner is compared with closed-world Bayes-adaptive planning that excludes the outside hypothesis, a
full-belief planner forbidden to defer, MAP certainty equivalence, persistent posterior sampling, best open loop,
myopic control, and immediate deferral. The primary mechanism gates require sensing at the root, deferral after a
green calibration result, continued sensing by the closed-world planner on that same history, reachable use of both
repairs, and material value over forced commitment, MAP, posterior sampling, open loop, and immediate deferral.

## Boundary

All probabilities and priors are explicit project-authored values, not estimated from LLM ranks. This is an exact
mechanism oracle, not empirical evidence that a real language model has these likelihoods or that a human supplies
these answers. It reads no language or model output, runs no model/API/training, mutates nothing, calls no service,
and performs no action or execution. A pass permits only a separate fresh-source design.
