# V223 archived semantic-adjudication metadata census results

## Bottom line

V223 found one workflow that is eligible for a more specific acquisition study: **Mondo structured term requests**.
Mondo passed all 17 noncompensatory metadata and provenance dimensions. Gene Ontology, Schema.org, and Wikidata were
useful partial matches, but each lacked at least one explicit outcome or adjudication contract required by the frozen
protocol.

The frozen branch is:

> `SOURCE_SPECIFIC_ACQUISITION_DESIGN_ELIGIBLE`

This is an eligibility result, not a language-understanding result. It authorizes a separately locked, Mondo-only,
metadata-first record/disposition census. It does not authorize opening request text, running a local or API model, or
treating repository labels as semantic gold without record-level validation.

## What was tested

The prospective V223 protocol compared four public semantic-change workflows:

1. Mondo GitHub term requests;
2. Gene Ontology GitHub term requests;
3. Schema.org GitHub proposals; and
4. Wikidata property proposals.

A source could pass only by satisfying every mandatory dimension: official authority, frozen revision and reuse
terms, real requester language, structured requests, independent human review, explicit accepted-new,
duplicate-existing, insufficient-evidence, and unsupported outcomes, versioned catalog consequences, recoverable
pre-request state, stable grouping, usable scale, separable language and outcome metadata, and non-model gold.

The formal run retrieved exactly the 21 frozen metadata, documentation, template, label, release, and category URLs.
All 21 succeeded, all snapshots matched their recorded hashes, and all true assessments cited captured evidence.
The snapshots totaled 549,788 bytes. No issue, proposal, comment, pull-request, discussion, archive record, request
title, or task body was requested.

## Source results

| Workflow | Dimensions passed | Missing dimensions | Eligible |
|---|---:|---|---|
| Mondo term requests | 17/17 | None | Yes |
| Gene Ontology term requests | 9/17 | Independent adjudication, public review, four outcome types, release linkage, human-gold contract | No |
| Schema.org proposals | 13/17 | Structured request schema and three explicit non-accept outcome types | No |
| Wikidata property proposals | 14/17 | Distinct duplicate, insufficient-evidence, and unsupported/out-of-scope outcomes | No |

Mondo qualified at the workflow level because the frozen official materials connect a structured user request to
community review, curation-team review, explicit workflow states, an assigned term identifier, a pull request, and a
monthly versioned release. The documented labels and policies make accepted-new, duplicate-existing,
clarification/blocked, and out-of-scope distinctions expressible.

The other sources are not judged low quality. They simply do not expose the complete four-way semantic disposition
contract needed for this study under frozen, machine-auditable metadata. Free-form closure or elapsed consensus was
not reinterpreted as semantic insufficiency or unsupportedness.

## Scientific meaning

V223 answers the earlier human-review concern in a bounded way:

> New live reviewers may not be necessary for a study of structured ontology-change requests if archived records
> demonstrably contain independent human requester and curator events with versioned catalog consequences.

It does not support replacing people with simulated judgments. The independent evidence would come from the archived
workflow itself. Nor does it generalize from specialist term requests to spontaneous conversational intent, where
the speaker's intended boundary may never be adjudicated or recorded.

## Required successor

The next stage must remain metadata-only. Before any Mondo request body can be opened, it must freeze and verify:

- an exact source revision, time window, and permanently excluded identifiers;
- mutually exclusive operational mappings for accepted-new, duplicate-existing, clarification/blocked, and
  unsupported/out-of-scope records;
- minimum usable counts in every required stratum;
- requester identity separated from a human reviewer, approver, or merger identity;
- accepted-record linkage from issue to merged pull request, new term identifier, and first containing release;
- group keys and leakage-resistant roles; and
- exclusion of AI-only, automated-only, ambiguous, reopened, or otherwise non-independent adjudications.

If those facts cannot be established from record metadata without request language, the branch stops. A later
language-opening protocol is conditional on that audit; a model condition is still further downstream and would be
allowed only for a deterministic residual.

## Preserved boundaries

- Formal task-record body reads: 0.
- Issue/proposal/comment/pull/archive record requests: 0.
- Protected research-record reads: 0.
- Local-model loads and generations: 0.
- API calls and training runs: 0.
- Ontology registration, trusted-state mutation, service actions, external side effects, and execution: 0.
- Previously protected Mondo JSONLs remain sealed.

