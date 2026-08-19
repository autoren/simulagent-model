# Prospective language pilot V1 — Phase 2 development plan

## Status and claim boundary

Phase 1 completed with 16/16 immutable requests, zero unable dispositions, and zero assistant generations. The
independent bundle audit passed. The request distribution was then inspected to check whether the participant had
entered requests rather than solutions. Consequently, Phase 2 is explicitly a **post-collection-frozen development
run**, not a blind confirmation.

The admissible claim is narrow:

> One frozen, bounded local assistant condition can be evaluated on 16 prospectively authored requests from one
> speaker for structured semantic uncertainty, conservative routing, truncation, and later participant-rated
> clarification utility.

It cannot establish unrestricted understanding, population reliability, ontology correctness, real-world competence,
authority, action, or execution.

## Why this condition

The initial requests contain a useful mixture. Some ask for general knowledge or drafting help that an offline model
may be able to provide. Others ask about local, private, live, or source-specific facts the assistant has not observed.
The central test is therefore not merely answer quality. It is whether the system distinguishes:

1. one actionable reading with adequate evidence;
2. multiple plausible readings or evidence the participant can supply; and
3. evidence requiring unavailable access or not obtainable from the participant.

## Frozen assistant architecture

The local model is a non-authoritative semantic proposer. It returns a candidate-goal set, a finite missing-evidence
code set, an evidence-source class, and exactly one compatible surface payload. A deterministic controller—not the
model—maps that structure to `PLAN`, `CLARIFY`, or `DEFER`.

- `PLAN` is possible only for one clear reading with no missing evidence.
- `CLARIFY` is possible for multiple readings or evidence the participant can supply, with one or two questions.
- `DEFER` is required for unavailable external evidence, unobtainable evidence, or invalid model output.

All plans are sandbox-only text. The model and controller have no execution tools.

## Model and decoding

- Local pinned `mlx-community/Qwen3.8-27B-4bit` revision
  `3e6447f082e89cc7f0bc6e5441afd38dfce760ff`.
- Temperature zero, one sample per request, no retries.
- Thinking enabled at `low`, hard-capped at 48 tokens.
- The runner mechanically closes reasoning and reserves a separate 320-token final continuation.
- Reasoning text is never persisted or shown.
- Parsed assistant messages must be persisted because the participant will later answer the actual frozen questions.

## Exposure and participant interaction

The run receives only the public Phase 1 fields: opaque record ID, initial request, and limited assistant-visible
context. It never receives the private goal, scenario facts, research metadata, or future participant answers.

If the run passes every mechanism and access gate, the Streamlit app may open a single clarification batch. It shows
only `CLARIFY` records, their original private scenario cards, locked requests, and the exact controller-approved
questions. It does not show candidate goals, reasoning, evidence codes, shadow outputs, or terminal plans.

The participant answers every clarification record before any terminal continuation is generated. No second
clarification wave is allowed in V1.

## Acceptance gates

- All 16 locked requests processed once.
- At least 15/16 raw outputs satisfy the exact schema.
- Every request receives a deterministic controller route.
- No final continuation reaches its token limit.
- No clarification contains more than two questions.
- Zero retries, API calls, service calls, side effects, or executions.

Failure freezes a negative development result and does not automatically expose questions to the participant.

## Later evaluation

After clarification and terminal continuation, the participant will judge goal understanding, likely task success,
constraint violations, clarification utility, and defer appropriateness. These participant judgments—not the model's
self-reported confidence—are the primary semantic evidence.
