# V89 Model-Free Failure Decomposition Results

V89 confirms that V88r1 cannot be rescued by constrained JSON or serialization repair alone. The
analysis used only frozen identifier-level fixture artifacts. It accessed no source utterance or prompt,
loaded no model, called no API, trained nothing, and performed no service call or side effect.

Among the 36 conforming rows, only eight had both the exact intent set and exact accumulated state-key
set. Nineteen had the intent right but state wrong, two had state right but intent wrong, and seven missed
both. The twelve malformed rows were all Flights cases: eight genuine `NONE` and four active-intent
records.

V89 then granted an unrealistically favorable serialization oracle: every malformed row received its
gold intent and state candidates, while every conforming row remained unchanged. Even this upper bound
failed two original semantic gates. Active-intent coverage reached only `0.7917` against the registered
`0.90`, and exact state-key sets reached `0.4583` against `0.50`. Intent-set exactness would rise to
`0.8125`, and state recall to `0.9208`, but those improvements are insufficient.

Replacing state with an oracle in addition to perfect serialization still left active-intent coverage at
`0.7917`. Therefore the next limitation is not merely output syntax or state accumulation: conforming
predictions also omit the correct active intent too often. State difficulty grows sharply with accumulated
cardinality; exactness was nearly absent for five or more state keys.

Freeze V89 with the decision to pause external local-model integration. Do not introduce constrained
decoding as if it solved the interface, do not build a state tracker solely to preserve this model role,
and do not use an API fallback, larger-model substitution, adapter, prompt edit, threshold relaxation, or
rerun without a genuinely new preregistered research question. The exact Bayesian core and deterministic
V83/V86 language boundary remain the validated system; an LLM remains optional and non-authoritative.
