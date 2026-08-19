# V90 local capacity and generation comparison

## Question

V88r1 and V89 established that the pinned Qwen3.5 4B proposer was not reliable enough to define the typed candidate set. V90 asks whether that failure is primarily associated with model capacity, model generation, or 4-bit quantization while leaving the deterministic validator, operational `NONE` state, Bayesian planner, and execution boundary unchanged.

## Fresh-source rule

The exposed V88 population is not reused. Before any new dialogue payload access, V90 pins `dev/dialogues_002.json` from the same official Schema-Guided Dialogue repository revision by byte size and Git blob SHA-1. Code may then produce a text-free structural inventory. No utterance may be printed or manually inspected before a later population lock.

## Intended independent model conditions

Subject to a positive source inventory and a separate frozen experiment design, the intended sequential local-only conditions are:

1. `mlx-community/Qwen3.5-4B-4bit`, the historical-family baseline;
2. `mlx-community/Qwen3.5-27B-4bit`, the same-generation capacity condition;
3. `mlx-community/Qwen3.8-27B-4bit`, the fixed-size generation condition;
4. `mlx-community/Qwen3.8-27B-8bit`, the quantization quality ceiling.

Every condition must use the same sealed records, prompt, output contract, deterministic temperature-zero decoding, parser, and scoring code. Models run independently and sequentially. Candidate unions and cascades are diagnostic-only and may be considered only after independent errors are frozen.

## Claim and authority boundary

This branch is offline human-language shadow evaluation. Model outputs are proposals only. They cannot select an action, update an authoritative belief, call a tool or service, or execute a side effect. No API model, adapter training, prompt tuning on evaluation records, manual source-language adjudication, or retry is authorized.

The source extension stage authorizes only one integrity-checked download and structural inventory. Model identities, gates, population, and inference are locked separately after the source result is known.
