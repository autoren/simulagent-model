# V223 archived semantic-adjudication metadata census plan

## Purpose

V221/V221r1 closed the retrospective Mondo release-reconstruction branch with a zero model-eligible residual. That
result cannot establish what a new speaker intends. V223 asks a different question: whether an existing public
semantic-change workflow contains archived, independently authored human requests and curator/community adjudication
that could support a bounded B2c study without collecting new human judgments.

This is not another ontology-release census and not another behavioral-abstention benchmark search. V215 already
established that versioned ontology artifacts can support retrospective reconstruction; V208 found no suitable paired
behavioral abstention dataset. V223 instead evaluates proposal and change-request workflows that may connect:

```text
requester language
    -> catalog-relative semantic proposal
    -> independent public review or consensus
    -> accept / existing-duplicate / insufficient / unsupported disposition
    -> versioned catalog consequence
```

## Frozen source units

The formal census will evaluate four workflow families:

1. `MONDO_GITHUB_TERM_REQUESTS` — structured Mondo term requests, curation documentation, public labels, pull-request
   review, and release linkage.
2. `GO_GITHUB_TERM_REQUESTS` — Gene Ontology new-term request templates, editor repository, labels, and versioned
   release metadata.
3. `SCHEMAORG_GITHUB_PROPOSALS` — public Schema.org proposal discussion, steering/community review, issue labels, and
   formal vocabulary releases.
4. `WIKIDATA_PROPERTY_PROPOSALS` — structured property proposals, proposer/creator separation, consensus policy,
   terminal proposal statuses, and revisioned property creation.

The URLs are frozen before the formal capture. GitHub documentation and templates are pinned to exact repository
commits where possible. Wikidata workflow pages are pinned by `oldid`. Dynamic repository-label, tag, and category
metadata will be stored and content-hashed at the frozen retrieval date.

## Claim and evidence dimensions

A source-specific acquisition design is eligible only if one source satisfies every noncompensatory dimension:

- official authority;
- immutable or content-hashed revision;
- compatible reuse terms;
- real requester-authored language;
- structured request schema;
- a reviewer/curator role distinct from the requester;
- public review or consensus;
- an explicit accepted-new outcome;
- an explicit existing/duplicate outcome;
- an explicit insufficient-evidence or clarification outcome;
- an explicit unsupported or out-of-scope outcome;
- accepted changes linked to a versioned catalog;
- recoverable pre-request catalog state;
- stable record provenance and grouping;
- credible scale across multiple disposition strata;
- separability of request language from outcome metadata; and
- human adjudication rather than model-generated gold.

Missing, ambiguous, merely conventional, or free-text-only evidence scores false. General availability does not imply
reuse permission. A closed issue is not automatically a rejection, and catalog absence is not automatically novelty.

## Language firewall

V223 may retrieve only repository metadata, workflow documentation, issue templates, label vocabularies, release/tag
metadata, Wikidata policy/template/module pages, and category counts. It may not request an issue, proposal, comment,
pull-request, discussion, or archive record endpoint; enumerate record titles; or inspect any task record body.

Discovery searches exposed a small number of snippets and example identifiers before this design was frozen. Those
identifiers are explicitly listed in the config and permanently excluded from any successor population. V223 makes no
blind-source-discovery claim. Formal task-record body reads remain zero.

## Branch rule

If at least one source satisfies all dimensions, V223 authorizes only a separately frozen source-specific acquisition
and identifiability design. That successor must first inspect structured record metadata and disposition counts without
request language, freeze exact outcome mappings and exclusions, and prove enough group-disjoint strata. Request bodies
may be opened only under a later lock.

If no source qualifies, B2c remains externally unvalidated. The project may continue retrospective reconstruction or
model-free mechanism studies, but it must not fabricate reviewers, simulate semantic validation, or use an LLM as its
own gold source.

## Preserved boundaries

- V220 protected JSONLs remain sealed.
- No earlier protected language or task records may be opened.
- No local or API model may be loaded or run.
- No training, ontology registration, trusted-state mutation, service action, external side effect, or execution is
  authorized.
- A positive V223 result would support only structured semantic-change requests, not unrestricted conversational
  open-world understanding.

