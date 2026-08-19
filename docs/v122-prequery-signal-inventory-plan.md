# V122 Pre-query Signal Inventory Plan

V122 inventories signals defined by frozen source code and configuration. It records when each signal is
available, whether it depends on the LLM proposal, whether it is deterministic, and whether it is eligible
for a future paired evaluation. It explicitly excludes hidden labels, post-query observations, and outcomes.

The inventory reads definitions only—no benchmark rows, language, generated responses, or metrics. A pass
can authorize only preregistration of a fresh model-free retrieval-geometry design. It cannot claim that any
signal predicts query value or that computational independence implies statistical independence.
