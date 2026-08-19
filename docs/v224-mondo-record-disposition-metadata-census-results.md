# V224/V224r2 Mondo record/disposition metadata census results

## Bottom line

The workflow-level promise found in V223 did not translate into a usable record-level four-way population. Across
2,397 Mondo issues in the frozen 2021–2024 creation window, none satisfied even a preliminary accepted-new,
duplicate-existing, clarification-needed, or out-of-scope record contract. The frozen branch is:

> `MONDO_B2C_EXTERNAL_VALIDATION_INSUFFICIENT`

V225 language acquisition is not authorized. Request titles and bodies remain unopened, and the four-way taxonomy is
not relaxed after seeing the result.

## Prospective design

V224 froze before record metadata access:

- 48 calendar-month search slices from 2021-01-01 through 2024-12-31;
- a 2025-12-31 event cutoff and exclusion of records updated in 2026 or later;
- permanent exclusions for issues 503, 673, and 10448;
- exact human-independence and AI/automation exclusions;
- four noncompensatory disposition mappings;
- SHA-256 selection of at most 64 records per preliminary stratum; and
- minimum preliminary and final counts of 24 per stratum.

The GraphQL selection omitted issue and pull-request titles, bodies, comments, review text, commit messages, and
rendered content. It retained only identifiers, timestamps, states, actors, labels, duplicate events, closing-PR
counts, and other structured metadata required by the protocol.

## Implementation repairs

The first capture stopped before any record response because the CLI could not separately encode a GraphQL document
and a same-named variable. V224r1 repaired that transport only, but the original monolithic query then received HTTP
502 before persisting any census artifact because it requested deep pull/review/file metadata for every issue.

V224r2 implemented the already-frozen preliminary-then-deep sequence: a thin safe enumeration first, followed by the
original full safe node query only for hash-selected records if the preliminary gate passed. The source frame,
selected field meanings, outcome rules, sample seed, thresholds, and decision rule did not change. Both failed
attempts are preserved in the provenance ledger; neither exposed or persisted task language.

## Census result

The completed thin enumeration retrieved exactly one page for each of 48 slices:

| Metric | Result |
|---|---:|
| Unique metadata records | 2,397 |
| Final `New term request` records | 362 |
| Records with an `approved` label event | 0 |
| Records with a `MarkedAsDuplicateEvent` | 0 |
| Final `duplicate` label records, all issue types | 13 |
| Final `needs clarification` label records, all issue types | 2 |
| Final `non-human animal` label records, all issue types | 20 |
| New-term requests with a closing-PR reference | 295 |
| Persisted task-language fields | 0 |

The 362 final new-term requests divided into 360 records with no single substantive outcome signature, one record
whose adjudicator did not meet the independent-human rule, and one record outside the requester rule. The broader
label counts already put duplicate, clarification, and non-human categories below the fixed minimum of 24, even before
requiring overlap with a new-term request, human event attribution, or catalog provenance.

Consequently, every preliminary stratum had count zero. The preregistered short-circuit fired, so V224 made no deep
pull-file, canonical-issue, or release-asset requests. This avoided mining prose or catalog artifacts after the count
failure.

## Interpretation

V223 and V224 answer different questions:

- V223: does the documented Mondo workflow make all required states conceptually expressible? **Yes.**
- V224: are those states recorded consistently enough in historical structured metadata to build the frozen study?
  **No.**

Many new-term issues have closing pull-request references, so accepted changes probably exist. What is missing is the
machine-auditable outcome annotation required to separate acceptance from routine closure while also supporting the
other three semantic strata. The current `approved` and clarification label vocabulary may be recent, inconsistently
applied, removed at closure, or used outside the frozen period; V224 does not distinguish those explanations because
doing so would require reading record language or changing the window after seeing counts.

This is not evidence that Mondo curation lacks human judgment. It is evidence that the archived structured metadata,
under this prospective contract, cannot substitute for a new expert-reviewed B2c dataset.

## Research consequence

B2c returns to externally unvalidated status. Existing ontology artifacts remain valid for retrospective
reconstruction, candidate generation, and model-free mechanism studies, but they cannot establish new-speaker intent.
Future B2c work requires either:

- a different source with explicit per-record adjudication states and immutable language snapshots;
- a prospective human/expert annotation workflow; or
- a narrower, separately preregistered claim that does not pretend to cover the missing semantic distinctions.

No local or API model should be run on these records: there is no defensible four-way gold population against which to
evaluate it.

## Preserved boundaries

- Task title/body reads and persistence: 0.
- Comment/review-text reads and persistence: 0.
- Protected research-record reads: 0.
- Model loads, generations, and API calls: 0.
- Training, ontology registration, trusted-state mutation, service actions, side effects, and execution: 0.
- Deep provenance requests after the preliminary gate: 0.

