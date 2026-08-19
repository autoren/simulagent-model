# V82 Local Clarification-Surface Plan

V82 moves the frozen local model out of semantic interpretation. The exact V79 planner first chooses
one of three typed clarification codes. Only then may the model propose surface wording for that code.
It never sees the original request and cannot change the code, candidate set, belief, action, observation
kernel, tool, or execution certificate.

Each code has exact lexical anchors. A deterministic validator requires those anchors exactly once,
forbids anchors belonging to other codes, rejects execution claims and extra fields, and checks a small
structural contract. Rejection discards the complete model output and substitutes a frozen canonical
question. No repair prompt or model retry is allowed.

The prospectively sealed population contains 24 code-and-style pairs, eight per code. The primary raw
model estimand is semantic acceptance before fallback. The deployed estimands are semantic validity,
action-code preservation, and V79 policy-value invariance after fallback. Canonical and finite-grammar
renderers are positive controls; unsafe mutations and raw passthrough are boundary controls.

The same pinned local MLX snapshot is used once per record with temperature zero and thinking disabled.
No API, adapter, human record, original user language, tool call, or external side effect is permitted.
Passing V82 would freeze only an optional wording layer. It would not rehabilitate V80/V81 or authorize
the model to interpret language, assign probabilities, choose clarifications, or execute actions.
