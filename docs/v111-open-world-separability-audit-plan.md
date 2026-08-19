# V111 Existing-Evidence Novelty Separability Audit Plan

V111 asks whether V110 failed because it used only nearest-neighbor score, or because the current frozen
single-turn evidence is not cleanly separable by simple deterministic rules. It reuses V110's exact
64/64 calibration/evaluation membership and generates no new model output.

The preregistered features are per-intent and per-scenario maximum character-TFIDF scores, top-two margins,
the score assigned to the LLM-proposed intent, frozen LLM status and confidence, and exact disagreement
between the proposed and nearest intent. Ten fixed axis-aligned rule families are enumerated over frozen
threshold grids. One rule is selected using calibration labels only, with joint novelty precision, recall,
and non-novel false-positive constraints preferred before F1 and deterministic tie breaks.

The selected rule is evaluated once. An evaluation-label oracle census is also reported only as a diagnostic
upper bound over the already registered family; it can never authorize a policy. No individual feature,
prediction, identifier, utterance, or raw model response is persisted. A transferred selected-rule pass
would permit only a separate full-policy development protocol. A selected-rule failure with zero evaluation-
oracle feasible rules closes the current evidence interface for this simple deterministic family. Nothing in
V111 authorizes protected-test access, schema induction, sequential planning, APIs, training, or execution.
