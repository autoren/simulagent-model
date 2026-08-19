# V88 External Intent-Candidate Shadow Plan

V88 is the first sealed evaluation of the structured-LLM boundary on independently authored human
language. It does not replace the exact Bayesian core and does not give language model output any
belief, action, service-call, or execution authority. The model may only propose a bounded typed
candidate set that remains permanently non-deployable.

The frozen V87 structural index is used to select 48 SGD user turns before any utterance is extracted.
Selection is deterministic under a registered SHA-256 salt and seven fixed service/intent strata. The
population contains 24 active-intent turns and 24 genuine `NONE` turns. Active cases are balanced across
the two flight intents, restaurant reservation, and ride sharing; `NONE` cases include every available
ride-sharing negative plus fixed flight and restaurant counts. No selected record can be replaced after
language extraction.

For each selected record, code extracts the complete dialogue history through the current user turn and
the target service's pinned schema. The frozen local MLX model receives service, intent, and slot
descriptions and must return exactly two JSON lists: a small intent candidate set that always retains
`NONE`, and accumulated state slot keys without values. The target intent set is `{NONE}` for a genuine
`NONE` state and `{active_intent, NONE}` otherwise. The target slot-key set is read directly from the
source state annotation.

Exhaustive enumeration, `NONE`-only, empty-state/gold-intent, and oracle controls expose trivial recall
and precision shortcuts. Success requires exact parsing and ontology conformance, active-intent coverage,
useful candidate-set precision beyond enumeration, state-key recovery, per-service recall, and zero API,
training, manual language inspection, service calls, or side effects. There is one deterministic local
generation per record, no malformed-output retry, and no prompt tuning after the corpus is sealed.

All source and derived language artifacts retain SGD attribution and CC BY-SA 4.0 status. Raw language
must never be printed by the builder or runner. Regardless of outcome, V88 remains an offline shadow
study and cannot authorize direct action, execution, an API dependency, or human-language deployment.
