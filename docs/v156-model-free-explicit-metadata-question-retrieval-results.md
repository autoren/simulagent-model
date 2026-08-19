# V156 model-free explicit-metadata question-retrieval results

## Verdict

V156 is a positive synthetic model-free development result. The frozen explicit-metadata retrieval policy selected the registered discriminating question first on all 96 development requests and matched the oracle sequential policy exactly.

Request-level top-1 accuracy and MRR were both 1.0, mean correct-query rank was 1.0, and the top-score tie rate was zero. All 96 ranks were first. The mean top-two score margin was 16.7891; even the smallest margin was positive. Each of the four request stages was independently 100% top-1, including unfamiliar paraphrases and within-family ambiguous requests.

Across 120 sequential episodes, retrieval mean cost was 0.3 and improvement over safe no-query abstention was 0.7. Oracle order had the same values. Source order had mean rank 3.5 and cost 1.05; seeded random had mean rank 3.35 and cost 1.005.

Every question-asking comparator reached the exact final state after the trusted closed answer. All 582 irrelevant intermediate question events from source and random ordering failed closed, the complete authoritative hypothesis universe was retained, candidate-state proposal fields were absent, and execution was zero.

Freeze the decision:

`explicit_metadata_retrieval_qualifies_on_synthetic_development_retain_model_free_policy_and_authorize_fresh_hard_tie_population_design_only`

## Interpretation

The result shows that the local LLM is unnecessary for this narrow routing problem when the registered catalog explicitly covers the request vocabulary. A simple deterministic policy was more accurate, cheaper, fully reproducible, and easier to verify than V154's local model:

- V154 direct: 83.333% top-1, MRR 0.9167, cost 0.34;
- V154 bounded low reasoning: 81.25% top-1, MRR 0.8938, cost 0.375;
- V156 deterministic retrieval: 100% top-1, MRR 1.0, cost 0.3.

This is not evidence that lexical retrieval solves open-world language understanding. The population is project-authored synthetic development language, and its explicit profiles contain vocabulary deliberately chosen to describe each question family. The perfect result demonstrates catalog-coverage feasibility and exposes the next missing condition: cases where two questions have equal or near-equal lexical evidence, where the request uses uncatalogued paraphrases, or where the correct distinction is relational rather than topical.

## Access boundary and successor

The preregistered policy performed 576 deterministic query scores. It read only frozen development projections. It did not read or score V155 evaluation, load a model or tokenizer, generate or score model output, call an API or service, train, induce capabilities, act, or execute.

Do not tune or rerun V156, add terms, change weights, open V155 evaluation, or introduce an LLM on this population. The next branch may only design a wholly fresh synthetic hard-tie population with prospectively defined strata:

- lexically decisive controls;
- uncatalogued but independently specified paraphrases;
- equal or near-equal topical evidence requiring a relational distinction;
- truly insufficient language that should not privilege any query.

That design must first compare deterministic retrieval, safe source/fallback order, and oracle order model-free. Only a separately justified later protocol may test a local LLM as a permanently non-authoritative tie-breaker. Trusted answers remain the sole semantic authority; model/API dependence, capability induction, action, and execution remain closed.
