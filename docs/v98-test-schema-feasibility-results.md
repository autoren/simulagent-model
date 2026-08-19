# V98 SGD Test-Schema Feasibility Result

## Verdict

V98 is a clean negative schema-feasibility result. The official SGD test schema contains 21 versioned
services across 18 service families, but only two families are absent from the complete development
schema: `Payment` and `Trains`. That is insufficient for a benchmark requiring three fresh catalog
families plus one fully withheld unsupported family.

The schema gate failed, so V98 stops before any test dialogue payload access, population selection,
language extraction, manual inspection, or model access.

## Result

The pinned 54,864-byte schema matched its Git blob identity. Family-level comparison against all 16
development families found two eligible typed services:

- `Payment_1`: two intents and four slots;
- `Trains_1`: two intents and ten slots.

Both meet the typed-service requirements, but the preregistered minimum was four eligible services from
four novel families. The novel-family and eligible-service gates therefore failed. The 34 test dialogue
shards remain unopened.

## Interpretation

Version suffixes do not create meaningful domain freshness. Counting `Banks_1` as novel after exposure
to `Banks_2`, for example, would inflate the open-set claim while reusing the same semantic family. The
family-level gate correctly prevents that.

Together, V97 and V98 close the complete official SGD repository as the sole source for this benchmark:
the development split is exhausted and the test split adds only two genuinely new families. This does
not invalidate the activation/current-turn construction validated by V95; it means a different source
dataset is required.

## Access and claim boundary

Only the pinned test schema was downloaded and automatically parsed. The structural inventory emitted
service and family identifiers plus counts, but no intent names, descriptions, slot names, tokens, or
dialogue language. Test-dialogue access, manual inspection, model loads, generations, API calls,
adapter training, real service calls, and external side effects were all zero.

V98 is schema-feasibility evidence only. It is not novelty, abstention, calibration, posterior, planning,
or execution evidence.

## Correct successor

Freeze V98 unchanged. The next source-selection stage must evaluate a different independently authored
dataset with at least four typed service/domain families, machine-checkable intent labels, utterance
language, and sufficient annotations for unsupported and insufficient-evidence controls. Candidate
sources must be audited for provenance, license, authorship, schema/slot structure, OOD construction,
and leakage before payload access. Synthetic language remains development-only evidence.
