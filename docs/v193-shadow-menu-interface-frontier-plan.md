# V193 shadow menu interface and oracle frontier plan

## Purpose

V193 fixes the exact contract between any future language component and the trusted clarification controller before a
single V192 utterance is scored. It also computes the proposal-quality frontier required to improve on V190's `0.38`
mean cost.

## Finite visible menu

The proposer sees 14 opaque option IDs with only domain and normalized intent-concept labels. Contract IDs and
historical truth kinds remain in a separate hidden map. This is a finite benchmark hypothesis menu, not unrestricted
open-world recognition and not an authority-granting ontology.

## Proposal grammar

A valid response is exactly one of:

```json
{"status": "RANKED", "ranked_option_ids": ["M01", "M02", "M03"]}
```

with one to three distinct valid option IDs, or:

```json
{"status": "INSUFFICIENT", "ranked_option_ids": []}
```

Missing, malformed, truncated, unknown, duplicate, extra-key, wrong-type, or free-text output maps deterministically
to `INSUFFICIENT`. There is no confidence field, retry, or repair generation.

## Trusted controller and costs

Two fixed policies consume the ranking:

- `TOP1_PLUS_OTHER`: a two-category trusted question, cost `0.10`;
- `TOP3_PLUS_OTHER`: a four-category trusted question, cost `0.20`.

If the target is present, the trusted answer identifies it exactly. If absent, trusted `OTHER` is followed by generic
clarification at cost `0.40`. Invalid or insufficient proposals use the unchanged V190 hierarchy. The proposal never
determines the terminal state or prunes the complete candidate universe.

Under an always-ranking abstraction, top-1 recall must exceed `0.30` merely to beat `0.38`, and must reach `0.35` to
improve by the required `0.02`. Top-3 recall must exceed `0.55` to break even and reach `0.60` for material value.
V193 verifies these thresholds on a fixed 0.001 recall grid and checks target-informed oracle controls.

## Decision

A full pass authorizes only a separately preregistered deterministic ranker evaluation on V192 development language.
It does not authorize immediate language scoring, any model/API use, protected access, ontology registration or
pruning, trusted mutation, action, or execution.
