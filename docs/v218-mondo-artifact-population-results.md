# V218 Mondo artifact population results

## Outcome

V218 is a clean negative result:

```text
NEGATIVE_MONDO_PAYLOAD_CONTROL_OR_POPULATION_FEASIBILITY
```

Exactly one preregistered gate failed. The release asset named `README.md` was a general repository README rather than
the published release-change summary represented by the GitHub release body. It therefore exposed only one of four
required summary categories under the frozen parser, for coverage `0.25` rather than `1.0`.

The project did not reinterpret the file, replace it with the already-seen API response, lower the category gate, or
rerun the population.

## Retrieval integrity

All nine exact assets passed retrieval integrity:

- Successful payload retrieval rate: `1.0`
- Total payload bytes: `98,957,852`
- Expected byte-count accuracy: `1.0`
- Published SHA-256 accuracy: `1.0`
- Raw hash coverage: `1.0`
- Unlisted network requests: `0`
- Remote import resolutions: `0`

The two Mondo base releases parsed as 35,997 older and 36,007 newer term stanzas with unique identifiers. Both source-
version controls and both obsoletion-candidate controls parsed. The new-term diff exactly matched the ten parsed
additions, and all 41 identifiers in the changed-term control belonged to parsed changed families.

## The failed control

The selected `README.md` asset contains repository-level material: project description, identifiers, download links,
editor instructions, citation, license, and contact information. It does not contain the release body's tables for
renamings, definition changes, and obsoletions with replacement.

The frozen category census found only `NEW_TERMS`, incidentally through generic prose about requesting new terms. It
did not find:

- `RENAMED_TERMS`;
- `TEXT_DEFINITIONS`; or
- `OBSOLETIONS_WITH_REPLACEMENT`.

This is a source-role mismatch, not evidence that those events are absent from Mondo. V217A's frozen GitHub release
metadata documented them, but V218 specifically required the downloaded release-summary control to contain them.

## Descriptive population evidence

The derived population passed every substantive event, identifiability, split, and consequence gate:

| Metric | Observed |
|---|---:|
| Eligible concept families | 385 |
| Paired evidence-mode records | 770 |
| Development groups | 289 |
| Protected groups | 96 |
| Distinct primary event types | 8 |
| Text-change families | 49 |
| Lifecycle-event families | 7 |
| Ambiguous unspecified families | 381 |
| Decision-contrast families | 385 |
| Semantic-state reconstruction | 1.0 |
| Version-space reconstruction | 1.0 |
| Boundary-witness coverage | 1.0 |
| Decision-consequence coverage | 1.0 |
| Cross-split overlap | 0 |
| Public source-ID leakage | 0 |

Primary event families comprised 10 additions, 18 definition changes, 71 asserted-logical changes, 251 mapping
changes, 15 name changes, four obsoletion-candidate status changes, three replacement changes, and 13 synonym changes.

These counts are descriptive boundary evidence. They do not override the failed noncompensatory control gate and do
not authorize deterministic method evaluation. The 96 protected groups are sealed.

## Scientific interpretation

V218 establishes two separate facts:

1. The selected Mondo pair has much stronger representational structure than the earlier Uberon pair: direct
   lifecycle events, many text changes, exact ambiguity under version-unspecified evidence, and complete separating
   witnesses.
2. An asset name is not a semantic role. GitHub's release `README.md` asset cannot be assumed to reproduce the GitHub
   release body, even when both appear under the same release.

The correct lesson is to freeze the release API body itself as a metadata control in a future untouched-pair protocol,
not to weaken the current gate.

## Claim and access boundaries

- V218 does not authorize V219 deterministic controls.
- The V218 development and protected populations may not be reused to repair this outcome.
- V218 protected records were created and checked structurally but were not used for method evaluation or manually
  inspected semantically.
- V216 and V213 protected data were not accessed.
- No local model, API model, or training was run.
- No ontology was registered and no trusted state, service, action, or external execution was invoked.
- The result concerns retrospective published Mondo states, not correct new-speaker intent.

## Correct successor

An in-place repair is not appropriate because changing the release-summary source would alter a preregistered payload
role after exposure. A successor may instead use a completely untouched adjacent Mondo release pair under a new
prospective lock. Its release-summary control should be the already-frozen official GitHub release metadata body, with
the selected release object and content hash fixed before any new payload retrieval.

Pair selection should be deterministic over the already-frozen V217A metadata, exclude both releases opened in V218,
and avoid inspecting any new ontology payload before the lock. If no untouched pair meets exact asset, event,
lifecycle, control, and byte gates, the Mondo payload branch should stop.

## Frozen artifact hashes before outcome verification

- Design: `7ef6365ca2bc7334aa8116ca88dd9c32188cb330def878276f50939e23bb6239`
- Design lock: `5f12495dcecfffa91a4a804d70ca63420dc4ab9faeb2eddbd01b594de281bf81`
- Retrieval manifest: `f56a3c0856e76f920d341432736308ca6eba1f98d252a8931647f5f309d14c3a`
- Parser control: `7c39f837d603cb7450fcf2fbf06df0e84d0dcf6cff8ec19fb757c629e2a6a9b1`
- Population manifest: `9b1f9550746e849ae8f19cc667b4ec87a28669cd082312c352894ab294275fa7`
- Summary: `3ac5b6230ddcf70b1a20ec01648276884e00e1f7acd05dcdebd4c7abcf5adf19`
- Result: `9413083a27c7d192228bc6edc9046eee06312c45297854e7d7d35a6cb560a196`
