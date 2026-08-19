# V80 Frozen Local-Model Candidate-Generation Plan

## Role of the model

V80 introduces a language model for the first time in this branch, but only as a
bounded candidate proposer. The model cannot assign probabilities, choose a
clarification, choose a tool action, certify execution, or perform a side effect.
Those responsibilities remain outside the model and ultimately belong to the
verified Bayesian decision layer.

The experiment uses the existing local MLX backbone
`mlx-community/Qwen3.5-4B-4bit` at revision
`0e7ffd5c629ef7719d4cbc04069232580bfa9d9c`, frozen and without an adapter. No API
is required or authorized.

## One-shot population

Twenty-four project-authored synthetic requests are frozen before inference:

- eight clear requests;
- four recipient-ambiguous requests;
- four operation-ambiguous requests;
- four fully ambiguous requests;
- four out-of-ontology requests.

The allowed interpretation vocabulary contains the four V79 concrete
interpretations plus `none_of_the_above`. Gold sets include every textually
plausible interpretation and always retain the escape hypothesis. This is a
controlled structured-language development benchmark, not human-language or
open-world evidence.

## Output and decoding

Each record receives one deterministic generation with thinking disabled,
temperature zero, and no malformed-output retry. The complete response must be
exactly one JSON object with the single key `candidate_ids`. Markdown fences,
explanations, probabilities, confidence, action fields, tool fields, duplicates,
unknown IDs, and noncanonical ordering all fail their registered checks.

The prompt, corpus, parser, scorer, model revision, and generation settings must
be sealed before the first forward pass. No prompt selection or resampling is
allowed on this population.

## Gates and next decision

The noncompensatory gates require perfect parsing, schema validity,
`none_of_the_above` inclusion, and canonical ordering; high overall and
per-stratum candidate recall; bounded overgeneration; and minimum exact-set
accuracy on clear and out-of-ontology strata. All 24 forward passes must remain
local, with zero API, adapter, human-record, real-tool, or external-side-effect
access.

Passing V80 would authorize only preregistration of a model-to-belief integration
stage. The model's token probabilities or self-reported confidence will not be
treated as calibrated beliefs. Failing V80 freezes this prompt and population;
any successor must use a newly preregistered prompt and fresh records.
