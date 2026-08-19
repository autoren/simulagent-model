# V144 pinned local certificate development realization

## Purpose

V143 proved that the frozen V142 certificate and deterministic finalizer can implement the desired policy under an oracle. V144 asks the next narrower empirical question: can the already pinned local Qwen3.8-27B model realize that certificate interface on the previously unused V142 development fixtures?

This is a single synthetic-development realization. It is not a test-split result, external transfer, unrestricted open-world understanding, deployment evidence, or permission for capability induction or execution.

## Frozen population and model

- Use exactly the 144 V142 development fixtures: 24 six-stage groups.
- Perform zero model generations on the 144 V142 test fixtures.
- Use `mlx-community/Qwen3.8-27B-4bit` at revision `3e6447f082e89cc7f0bc6e5441afd38dfce760ff`.
- Load the model at most once.
- Run exactly one deterministic thinking-enabled generation per development fixture, with at most 1,024 new tokens and no retry.
- Do not use an API, training, V134 language, external language, real services, tools, or execution.

## Frozen interface

The model does not emit the system's final action syntax. It emits a non-authoritative evidence certificate containing exactly:

```json
{
  "evidence_status": "SUFFICIENT or INSUFFICIENT",
  "compatible_choice_ids": ["sorted supplied non-A00 IDs"],
  "proposed_choice_id": "one supplied ID"
}
```

`SUFFICIENT` is valid only for a singleton compatible set whose member matches the proposal. `INSUFFICIENT` is valid only for at least two compatible choices and an `A00` proposal. The V138 stateful parser separates the prompt-opened thinking trace from the final JSON. The V142 validator checks the certificate. The deterministic finalizer maps every invalid, missing, truncated, unknown, or inconsistent certificate to a programmatically serialized `A00` output.

Only a valid normalized certificate, hashes, token counts, timing, and validation metadata may be retained. Raw responses, final text, and thinking traces must not be persisted or manually inspected.

## Noncompensatory evaluation

V144 separately measures:

- certificate structural validity and deterministic final-output validity;
- exact final classification overall and by language class;
- exact compatible-set recovery and retention of the hidden compatible options;
- ambiguity sensitivity, decidable specificity, and proposal correctness conditional on a sufficient certificate;
- six-stage group accuracy, false-known errors, and attraction to the fallible presented candidate;
- sequential query behavior, decision cost, improvement over not querying, and right-side safety;
- thinking-trace closure, access budgets, immutable authoritative hypothesis retention, and execution count.

All preregistered gates must pass. Formatting safety cannot compensate for semantic failure, and high aggregate accuracy cannot compensate for failed ambiguity, specificity, language-class, false-known, or sequential gates.

## Decision boundary

Passing authorizes only the design and preregistration of one separately frozen V142 test realization. It does not authorize opening or running the test split immediately. Failure closes this local certificate branch without prompt tuning, retries, a larger token ceiling, selective exclusion, result mining, or rerunning V144.
