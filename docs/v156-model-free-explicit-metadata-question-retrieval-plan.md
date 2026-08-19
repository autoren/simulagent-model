# V156 model-free explicit-metadata question-retrieval plan

## Question

Can a frozen deterministic lexical policy use visible request language and explicit registered-question metadata to order the useful trusted clarification question reliably enough to replace the local LLM on this narrow routing task?

V156 is deliberately not an open-world classifier. It cannot propose states, decide catalog membership, answer a clarification, define a capability, or act. Its only output is a permutation of registered question IDs. Incorrect ordering can increase interaction cost but cannot change the trusted final decision.

## Prospective policy

For each query, the policy counts exact normalized anchor phrases, primary terms, secondary terms, and tokens from the visible question/title/option text. Their fixed weights are 8, 3, 1, and 0.25. Scores are sorted descending with registered source order as the only tie-break. There is no fitting, learned parameter, threshold selection, state field, witness field, choice ID, truth label, or oracle input.

The preregistration step creates a state-free retrieval-catalog projection, a public development-request projection, and a language-free development metadata projection. It freezes their hashes before the policy is run. The runner may read only those projections plus the full catalog solely for deterministic trusted-witness routing after the correct question is eventually asked. It may not read the V155 evaluation split.

## Comparators and gates

The development policy is compared with no-query abstention, source order, seeded-random order, and oracle order. The retrieval condition must independently satisfy:

- at least 95% request-level top-1 accuracy;
- at least 0.97 request-level MRR;
- mean correct-query rank at most 1.10;
- top-score tie rate at most 5%;
- sequential mean cost at most 0.33;
- improvement over safe no-query abstention at least 0.67;
- exact final state after trusted answers;
- complete fail-closure for irrelevant intermediate questions;
- complete authoritative-hypothesis retention;
- zero candidate-proposal fields and zero execution.

Passing is synthetic model-free development evidence only. It authorizes design of a new hard-tie population, not use of an LLM hybrid on V155. Failure closes without weight changes, term edits, threshold fitting, evaluation access, or model fallback.

No model/tokenizer, API, training, induction, authority, action, side effect, or execution is permitted.
