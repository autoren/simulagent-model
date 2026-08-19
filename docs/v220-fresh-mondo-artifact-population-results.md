# V220 fresh Mondo artifact population results

## Outcome

V220 is a positive fresh-payload and population result. The frozen branch is
`FRESH_MONDO_REPRESENTATIONAL_POPULATION_ELIGIBLE`.

The experiment retrieved exactly the eight V219A-attested assets for `v2026-05-05 -> v2026-06-02`. All 96,095,320
bytes matched their preregistered byte counts and published SHA-256 digests. There was no unlisted request, release-body
request, or remote import resolution.

## Published controls

Both OBO releases parsed with unique identifiers: 31,885 older terms and 33,767 newer terms. The pair-specific control
contained exactly the 1,882 parsed additions, and every one of its 21 changed identifiers belonged to a parsed changed
family. Both candidate-status controls and both source-version controls parsed successfully.

The release-summary control was the already content-hashed official `v2026-06-02` API body. Its SHA-256 remained
`76b1133b7196e1547ca91470bcbb8e59422e8fe2fe815e7ebab250473d72050a`, and it covered all four required categories.
No README was used.

## Population

The deterministic builder produced:

- 2,161 role-separated concept families;
- 4,322 paired `VERSION_UNSPECIFIED` / `CURRENT_RELEASE_DECLARED` records;
- 1,621 development groups and 540 sealed protected groups;
- 1,882 addition-event families;
- 213 text-change families;
- 14 lifecycle-event families;
- 16 mapping-event families;
- 2,153 ambiguous version-unspecified families; and
- 2,161 families with a decision contrast between evidence modes.

There were eight distinct primary event types. Event strata were counted by family membership in any qualifying event,
so a large addition stratum could not compensate for missing text, lifecycle, mapping, ambiguity, or decision evidence.

## Integrity and oracle checks

Payload accounting, retrieval success, raw hashes, byte counts, published digests, tabular parsing, new-term agreement,
changed-term precision, and release-summary coverage were all exact (`1.0`).

Development-only semantic-state reconstruction, version-space reconstruction, boundary-witness coverage, and decision-
consequence coverage were all `1.0`. There was no group overlap, duplicate case identifier, public source-identifier
leakage, or public/truth mismatch. All metrics were finite.

The protected JSONL files were created by the frozen builder and then only hashed. Neither the runner nor the outcome
verifier loads protected records for scoring. Their method-evaluation, manual-inspection, and scoring-load counters are
all zero.

## Interpretation

V220 is fresh evidence that the representational-diagnosis construction is not peculiar to the V218 release pair and
that the V218 negative was specifically a release-summary provenance failure. It does not retroactively repair V218.

The evidence supports retrospective reconstruction of asserted ontology states and lifecycle consequences. It does not
establish inferred OWL equivalence, complete curator rationale, correct new-speaker intent, expert validation, or model
quality.

## Authorization

V220 authorizes design of one development-only V221 deterministic candidate/version-space residual study. Protected
evaluation and model use remain closed. A model can be considered only later if deterministic controls leave a
preregistered meaningful residual, and then only as a bounded candidate generator measured by incremental oracle-class
recall at fixed budget.

