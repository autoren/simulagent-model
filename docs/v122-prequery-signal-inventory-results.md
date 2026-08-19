# V122 Pre-query Signal Inventory Results

## Outcome

V122 passed as a static provenance result:

> `inventory_complete_retrieval_geometry_only_llm_independent_semantic_family`

It inventoried 11 pre-query signal definitions and excluded 9 sources that would leak hidden labels,
post-query evidence, outcomes, or additional model interrogation. It read no individual records or language,
loaded no model, evaluated no signal, fitted no trigger, and performed no execution.

The only semantic family that is computationally independent of the LLM is frozen retrieval geometry:

- nearest character n-gram similarity;
- nearest retrieved intent, for context rather than as ground truth;
- the status band derived from frozen retrieval thresholds.

The raw observation-presence flag is also model-free, but it is only a control. Direct choices and confidence
depend on the proposer. Validator outputs depend on a generated response. Agreement, score-gap, same-scenario,
and policy-state features all depend on the proposed candidate. They therefore cannot serve as independent
evidence that the proposal is reliable.

Computational independence is not a statistical-independence claim. Retrieval and the LLM may fail on the
same language. V122 also makes no claim that retrieval geometry predicts clarification value or supports a
selective-query policy.

The only newly authorized action is to preregister a fresh, model-free, paired retrieval-geometry design.
Signal evaluation, threshold fitting, language/model access, protected-set access, induction, API calls,
training, authority, and real execution remain closed.
