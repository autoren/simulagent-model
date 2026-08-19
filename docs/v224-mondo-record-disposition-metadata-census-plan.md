# V224 Mondo record/disposition metadata census plan

## Purpose

V223 established workflow-level eligibility for Mondo structured term requests. V224 asks whether individual archived
records expose enough structured metadata to form a clean, independently human-adjudicated four-way population before
any request language is opened.

The target distinctions are:

1. accepted as a genuinely new catalog term;
2. duplicate of a concept already represented at request time;
3. insufficient evidence requiring clarification; and
4. unsupported or outside Mondo's scope.

V224 is not a language benchmark. It is a record-provenance and outcome-identifiability census.

## Frozen population frame

- Repository: `monarch-initiative/mondo`.
- Workflow revision: `0056dc115d6d1e33a39075a6c699f661f7866478`.
- Request creation window: 2021-01-01 through 2024-12-31 UTC.
- Resolution/event cutoff: 2025-12-31 23:59:59 UTC.
- Search slices: the 48 calendar months in the creation window.
- Search constraint: issues must not have been updated on or after 2026-01-01, so current structured state is not a
  post-cutoff mutation of the frozen history.
- Permanently excluded issues: 503, 673, and 10448, which were exposed during V223 discovery.

The formal search selects only GraphQL fields explicitly allowlisted in the query module. Titles, bodies, body edit
content, comments, review text, commit messages, descriptions, reactions, and rendered text are absent from the query.

## Metadata firewall

Allowed record fields are stable identifiers, timestamps, state/state reason, actor type and login, label names,
label/unlabel/close/reopen/duplicate events, canonical duplicate links, closing pull-request identifiers, merge
metadata, review state and actor, changed-file paths and counts, and release asset metadata.

For selected accepted or canonical records, the capture worker may request pull-request file metadata. It must discard
patch text in memory and persist only response hashes, file paths, counts, and exact added `MONDO:` identifiers.
Official `mondo_release_diff_new_terms.tsv` assets may likewise be streamed through a worker that persists only the
asset hash, size, tag, publication time, and exact Mondo identifiers. No ontology labels or definitions are retained.

The following remain forbidden:

- issue or pull-request title/body fields;
- comments or review bodies;
- task/request language;
- existing protected research JSONLs;
- local or API model access;
- training, registration, mutation, service action, side effects, or execution.

## Exclusions

Exclude a record if any of these hold:

- requester is absent, a bot, an organization, or a known automation account;
- any AI/automation marker occurs (`ai-curation`, `ai-success`, `ai-failure`,
  `ai-needed-some-guidance`, `claude`, `Automated`, or `hallucination`);
- the relevant adjudication actor is absent, non-human, or identical to the requester;
- relevant timeline, label, pull-request, review, or file pagination is truncated;
- the issue was reopened after its proposed terminal disposition;
- multiple substantive outcome signatures conflict;
- the request body was edited after the decisive clarification or out-of-scope label;
- an outcome exists only in prose or is inferred from closure, catalog absence, or similarity; or
- accepted/duplicate provenance cannot be linked exactly to a term identifier and release.

## Noncompensatory outcome mappings

### Accepted new

Require a `New term request` issue, an independently human-applied `approved` event, terminal closure, a merged closing
pull request with an independently human merger, a changed `mondo-edit.obo`, exactly one added Mondo identifier in the
filtered patch, and first appearance of that identifier in an official new-terms release asset after the merge.

### Existing or duplicate

Require a human `MarkedAsDuplicateEvent` with an exact canonical issue. The canonical issue must itself satisfy the
accepted-new provenance chain, and its term must appear in an official release before the duplicate request was
created. A `duplicate` label without a canonical event is insufficient.

### Insufficient or clarify

Require a final `needs clarification` label applied by an independent human, an open issue at the cutoff, no competing
outcome signature, and no body edit after that label. `blocked` alone is insufficient.

### Unsupported or out of scope

Require a final `non-human animal` label on a new-term request, applied by an independent human, followed by terminal
human closure, no competing outcome signature, and no body edit after the scope label. The pinned Mondo scope policy
must support the mapping. Generic closure, `Candidate for Closure`, and `hallucination` do not qualify.

## Sampling and gate

After full metadata enumeration, records are ordered within preliminary strata by SHA-256 over the frozen seed and
issue number. At most 64 records per stratum receive deep provenance evaluation. At least 24 records must survive in
each substantive stratum. Every retained issue number is a distinct group.

If any preliminary stratum has fewer than 24 records, deep pull/release retrieval is skipped and the result is
negative. If deep evaluation runs but any final stratum has fewer than 24 independently adjudicated records, the
result is also negative. Counts cannot be pooled across meanings or repaired by reading language.

## Branch rule

- Pass: authorize only a separately frozen V225 role-separated language acquisition and identifiability design.
- Fail: freeze Mondo B2c as externally unvalidated under this structured four-way contract. Do not open language or
  relax the taxonomy after seeing the shortfall.

