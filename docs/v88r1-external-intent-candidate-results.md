# V88/V88r1 External Intent-Candidate Results

V88r1 is a clean negative external-human-language result. The original V88 attempt is separately frozen
as execution-inconclusive after one unobserved generation because of a missing harness identity field.
The terminal V88r1 retry changed only that field handoff, reused every scientific dependency byte-for-
byte, completed all 48 sealed records, and cannot be rerun.

The frozen local `mlx-community/Qwen3.5-4B-4bit` model did not satisfy the candidate-interface gates.
Exact JSON and ontology conformance were both `0.75`; mandatory `NONE` inclusion was `0.75`; active-
intent coverage was `0.625`; exact intent-candidate sets were `0.5625`; and genuine-`NONE` exactness was
`0.5833`. Mean state-slot-key recall was `0.6708`, while exact state-key sets were only `0.2083`.

The failure was domain-sensitive. Intent recall was `0.3684` for `Flights_3`, versus `0.9167` for
`Restaurants_2` and `0.9091` for `RideSharing_1`. All twelve malformed outputs occurred in the flight
service. Among the 36 ontology-conforming outputs, intent-set exactness rose to `0.75` and state-key recall
to `0.8944`, but active-intent coverage remained `0.75` and exact state-key sets only `0.2778`. Therefore
serialization accounts for part, but not all, of the failure.

The model did beat exhaustive intent enumeration on exact-set rate by `0.3958` and kept the mean intent
candidate count to `1.1875`, so the result is not evidence that the model extracted no useful signal.
It is evidence that this frozen local model and interface are not reliable enough to define the live
hypothesis set or accumulated state on independently authored dialogue.

All 48 outputs remained permanently non-deployable. Across the failed startup and terminal retry there
were two local model loads and 49 generations, with zero API calls, adapter training, manual utterance
inspection, live service calls, or external side effects. Source and derived language remain attributed
to the Schema-Guided Dialogue Dataset at the pinned revision under CC BY-SA 4.0.

Freeze V88/V88r1 negative. Do not repair formatting post hoc, change the prompt, replace records, relax
thresholds, call an API model, train an adapter, or rerun. The verified model-free V83/V86 interface and
Bayesian core remain the trustworthy path. A successor should first perform a model-free decomposition
of the frozen failures and, if justified, preregister a materially different factorization in which
deterministic code owns JSON structure and state accumulation while a local model proposes only a narrow
intent compatibility signal. It still may not select actions or execute services.
