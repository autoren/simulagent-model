# V201r2 repair-decision label verification results

V201r2 passes. V201r1 failed only because its stored `decision` field contained the intentionally preserved V201
scientific decision rather than the repair-stage label. Every substantive V201r1 elapsed-time repair check was true.

The stored source decision is exactly:

`freeze_V201_negative_or_presentation_sensitive_without_retry_reprompt_model_selection_or_API`

V201r2 records the repair-stage decision separately as:

`freeze_V201r2_serialization_repair_and_preserve_V201_negative_result`

No source artifact, model output, metric, gate, or scientific interpretation changed. Source mutation, model or
policy reruns, raw-response reads, APIs, and execution were zero. V201 remains a formal negative on stable top-3 set
invariance and does not authorize paired protected robustness.

