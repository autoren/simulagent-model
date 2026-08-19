# V89 Model-Free Failure Decomposition Plan

V89 uses the already frozen V88r1 identifier-only fixture artifacts to distinguish a serialization
failure from semantic intent and accumulated-state failures. It cannot read source utterances or prompts,
load a model, call an API, train, or execute anything.

The strict result is reconstructed first. The strongest serialization-only counterfactual then grants
every malformed row perfect gold predictions while leaving all conforming rows unchanged. This is an
optimistic upper bound, not a deployable repair: if even it misses an original semantic gate, constrained
JSON or post-processing cannot be sufficient. Separate intent and state oracles localize the residual
bottleneck. Joint exactness quadrants, per-service malformed counts, label roles, and state-cardinality
buckets are fixed before computation.

A positive audit only freezes the diagnosis. If serialization plus a state oracle clears the intent
gates, the next permissible action is a model-free feasibility study for deterministic state accumulation.
No result can authorize another model run, API comparison, adapter, prompt change, deployment, or service
execution.
