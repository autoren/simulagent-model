# V79 Explicit Terminal-Utility Successor Plan

## Purpose

V79 is a targeted model-free successor to the frozen V78 negative result. It is
not a rerun or a threshold relaxation. It inherits the complete V78 task family,
priors, evidence channels, rewards, action order, controls, and twenty gates.
The only decision-problem change is an explicit utility for reaching horizon
exhaustion while the task remains active and unresolved.

## Frozen semantic correction

- terminal state at horizon exhaustion: utility `0`;
- active unresolved state at horizon exhaustion: utility `-6`;
- explicit abstention: reward `-5`;
- safe preview: reward `-2.5`.

Thus silent unresolved expiry is worse than explicitly abstaining, while a safe
reversible resolution remains preferable to both. The terminal utility applies
to the exact policy and every control, including open-loop evaluation.

## Rationale

V78's unknown-heavy policy never executed unsafely, but at horizon one it asked
again because asking cost only `-1` and unresolved expiry cost `0`. The registered
safe-continuation gate correctly rejected that behavior. V79 represents the
previously omitted consequence rather than weakening the gate.

The value `-6` is fixed before any V79 policy computation. It is chosen from the
semantic ordering above, not by optimizing a V79 outcome: it is exactly one unit
worse than explicit abstention and remains far above the `-40` consequence of an
incorrect irreversible execution.

## Evaluation and stopping rule

The hardened durable census harness, fresh implementation audit, all inherited
V78 gates, and two terminal-utility gates are mandatory. Any failure is frozen
without modifying the inherited design or terminal value. All raw fixtures are
written before gate aggregation and independently recomputed afterward.

Passing V79 authorizes only preregistration of frozen local-model candidate
generation. It does not authorize a model forward pass, API access, adapter
training, human-data collection, learned likelihoods, or real-tool execution.
