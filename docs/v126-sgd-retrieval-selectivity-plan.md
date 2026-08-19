# V126 SGD Retrieval-selectivity Plan

V126 is the first cross-dataset test of the V122 signal family. It automatically extracts the already locked
4,881 train and 576 evaluation utterances from the pinned SGD archive, without persisting or exposing them.
The exact V106 character n-gram retriever and V112 thresholds are reused: similarity at least 0.8 skips to
the nearest known shadow action, similarity at most 0.3 skips to unsupported, and the middle band queries.

There is one trigger and no fitting or alternative selection. The frozen asymmetric causal clarification
channel is evaluated at 95% marginal correctness, three priors, and correlations 0, 0.25, and 0.50. Every
query costs 0.30. The primary policy must beat the 1.1667 ask-always regret, preserve known and unsupported
accuracy and false-known safety, query cases whose average decision value is at least the cost, skip cases
whose average value is no greater than the cost, and never be worse than query-always.

The full eleven-choice safe composite universe is retained. No LLM, API, protected set, manual language
inspection, capability induction, authority, or execution is permitted. A negative closes the currently
inventoried retrieval-status trigger rather than authorizing threshold tuning.
