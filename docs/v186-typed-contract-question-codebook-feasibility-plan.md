# V186 typed contract question-codebook feasibility plan

## Question

Can a finite set of typed, human-interpretable binary attributes distinguish the 14 frozen external capability contracts without using service version labels, source truth classes, presented candidates, utterance language, or similarity scores?

V186 is structural and text-free. It does not run a planner or claim that any person or model supplies the answers.

## Question vocabulary

Questions are generated exhaustively from the frozen contract payload:

- exact normalized intent concept membership;
- domain membership;
- presence of any service slot;
- required-slot membership;
- result-slot membership; and
- transactionality.

Invariant questions are removed. Every retained question has both answer values among the 14 contracts. Question order and identifiers are canonical. Description hashes, service-version labels, source-definition IDs, truth kinds, and language are forbidden inputs.

## Identifiability rule

Each contract receives its complete binary answer vector. Contracts with equal vectors remain one equivalence class; V186 may not force a hidden distinction. Full identification requires 14 singleton classes and exhaustive separation of all 91 contract pairs.

## Role binding

Every V183 development and protected opaque record is bound to the frozen vector of its target contract. Missing controls have no vector and remain `INSUFFICIENT`. Bindings are stored separately by role. Protected utterance language is neither needed nor read.

## Gates and boundary

The codebook must contain at least 14 nontrivial unique questions from at least three families, yield 14 unique vectors and singleton equivalence classes, separate all contract pairs, reconstruct every observed target vector, preserve exact 132/132 role counts, and use no policy scoring, language, model, API, training, registration, state mutation, service, side effect, action, or execution.

Passing authorizes only a separately preregistered clean exact-planner comparison under explicit costs. Failure freezes the observational equivalence classes and stops before planning.
