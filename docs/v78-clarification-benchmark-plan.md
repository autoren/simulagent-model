# V78 Fresh Clarification-and-Tool-Use Benchmark Plan

## Status

V78 is a fresh, exactly enumerable, model-free development benchmark. It is
prospectively specified after V77 and V77r1 were frozen as execution-inconclusive.
No V77 registered value or optimal action was persisted or emitted, and V78 uses
a new task family, new surface instructions, new priors, new observation
channels, and new rewards.

## Scientific question

Can an exact posterior-aware decision layer preserve multiple plausible
interpretations long enough to choose useful clarification or reversible preview
actions, while point-estimate controls either act too early or ask when they do
not need to?

The hidden interpretation combines two uncertain fields:

- operation: schedule a review or send a summary;
- recipient: Alex Chen or Alex Kim;
- plus an operational `none_of_the_above` hypothesis.

The irreversible actions execute one of the four concrete interpretations.
They are available to the exact planner only when the complete posterior meets
both the matching-hypothesis and none-hypothesis certification thresholds.
Reversible preview and abstention remain available without that certificate.

## Why this is relevant to an LLM interface

The intended later architecture is deliberately asymmetric:

1. a frozen LLM proposes a bounded structured candidate set from text;
2. calibrated evidence assigns weights without treating model confidence as
   ground truth;
3. the verified Bayesian core selects clarification, preview, execution, or
   abstention;
4. irreversible execution remains fail-closed on the complete belief state.

V78 tests only steps 2–4 with a frozen, project-authored hypothesis set. It does
not use a local model, API model, human record, real tool, or external side
effect. Passing V78 would authorize only a new preregistration for frozen local
model candidate generation.

## Registered fixtures

- `ambiguous_tool_intent`: operation and recipient are both ambiguous;
- `clear_tool_intent`: one interpretation already meets execution certification;
- `unknown_heavy_tool_intent`: `none_of_the_above` starts with majority mass;
- `dominant_safe_preview`: uncertainty is present but a reversible preview
  dominates under every hypothesis.

The last fixture is the boundary control: Bayesian machinery must not manufacture
an information-seeking advantage merely because uncertainty exists.

## Controls and gates

The exact policy is compared with MAP interpretation, persistent posterior
sampling, act-immediately, ask-always, best open-loop, and an interpretation
oracle. All values are computed exactly at horizon three.

Every registered gate in
`configs/v78-clarification-benchmark-design.json` must pass. No fixture, prior,
reward, threshold, horizon, action order, or gate may be changed after the design
lock. A failed scientific gate is a frozen negative result. An execution error
must preserve every completed raw fixture through the locked census harness and
does not authorize a rerun.

## Claim boundary

V78 can establish only a synthetic, model-free development mechanism. It cannot
establish natural-language understanding, LLM calibration, human usability,
external benchmark performance, open-world safety, or authority to execute
tools. Those require separately named, prospectively locked stages.
