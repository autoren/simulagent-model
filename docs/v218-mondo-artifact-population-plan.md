# V218 Mondo artifact population plan

## Question

V217A established that Mondo is the only one of three prospectively frozen external sources with exact bounded
releases, diverse published change events, lifecycle evidence, and plausible ambiguity. V218 asks:

> Can one exact pair of published Mondo base releases support a group-disjoint, artifact-grounded population for exact
> representational diagnosis, including evidence states, expressibility, version spaces, boundary witnesses, and
> decision consequences?

This is a payload and population-feasibility study. It is not a language-model experiment, a new-speaker intent study,
or an ontology-registration exercise.

## Prospective boundary

Before this lock, V217A inspected and hashed official release metadata. That metadata disclosed release names, asset
URLs, byte counts, declared SHA-256 digests, and published release-note summaries. No selected payload body, OBO term,
definition, axiom, TSV row, or formal population record was read.

V218 freezes nine exact requests totaling 98,957,852 expected bytes under a 99,000,000-byte ceiling:

- `mondo-base.obo` for `v2026-07-06` and `v2026-08-04`;
- the newer release's changed-term and new-term diff controls;
- both releases' obsoletion-candidate controls;
- both releases' source-version controls; and
- the newer release's published README summary.

Every asset has a GitHub-declared SHA-256 digest. Retrieval is one attempt per asset, without substitutions, link
expansion, or remote import resolution.

## Parsing and provenance

The parser reads `[Term]` stanzas as UTF-8 and fails on duplicate identifiers. It keeps provenance roles separate for
old source, current target, pair-diff controls, candidate-status controls, source-version controls, and release summary.

Normalized asserted state contains:

- text: name, definition, and synonyms;
- asserted logical fields: `is_a`, `relationship`, `intersection_of`, `equivalent_to`, and `disjoint_from`;
- asserted mappings: `xref`; and
- lifecycle fields: `is_obsolete`, `replaced_by`, and `consider`.

The resulting state hashes are exact equivalence classes only for these normalized asserted fields. They are not OWL
closure or inferred semantic equivalence.

## Events and concept families

The pair is differenced for additions, removals, name, definition, synonym, asserted-logical, mapping, obsoletion,
replacement, and obsoletion-candidate membership changes. A union-find groups stable identifiers with all asserted
`replaced_by` and `consider` links seen in either release. All versions, linked identifiers, records, and evidence modes
for one family remain in one split.

Each eligible family produces two records:

1. `VERSION_UNSPECIFIED`, where old, current, absence, and linked replacement states compatible with the surface
   evidence remain in the candidate set; and
2. `CURRENT_RELEASE_DECLARED`, where the current published state, asserted replacement, or non-expressibility controls
   the answer.

The public record contains only redacted surface text, release evidence, requested target, and evidence mode. Hidden
truth contains source identifiers, event types, exact state and version-space classes, expressibility, evidence state,
the first separating asserted-field witness, and an exact shadow decision consequence.

For unspecified evidence, forcing one class incurs unit loss whenever multiple states remain possible; preserving the
version space or clarifying has zero loss. With the current release declared, resolving the current state, following an
asserted replacement, or abstaining when not expressible has zero loss, while a mismatched lifecycle action has unit
loss. This is a deterministic consequence oracle, not yet a planner result.

## Controls and noncompensatory gates

The frozen new-term control must exactly match parsed additions. Every changed-term control identifier must belong to a
parsed changed family. Candidate and source-version TSVs must parse, and the release README must expose all four frozen
summary categories.

The population must contain at least:

- 24 eligible concept families and 48 paired evidence-mode records;
- four primary event types;
- 12 text-change families;
- three lifecycle-event families;
- 12 ambiguous unspecified families;
- 12 families whose correct decision changes when current-release evidence is supplied;
- 18 development groups and six protected groups.

State reconstruction, version-space reconstruction, boundary witnesses, decision consequences, digest checks, and
control agreement are exact gates. Cross-split overlap, duplicate cases, public source-ID leakage, remote import
resolution, and protected method/manual inspection must all remain zero. There is deliberately no total-term threshold.

## Split and outcome

Concept families are ranked by a stable SHA-256 key; every fourth family is protected. The builder may create and hash
the protected artifacts and verify their structure, but neither their semantic content nor any method result may be
inspected in V218.

A pass authorizes only a separately locked V219 deterministic-control design on development data. A local model is
ineligible until deterministic controls leave at least 12 genuine development residual groups under a separate lock.
An API model is not required.

Any failed payload, parser, control, population, identifiability, split, integrity, or access gate freezes a negative
V218 result. The project may not change payloads, parser rules, source, event definitions, split, thresholds, or gates
after exposure.

## Claim boundary

Mondo artifacts can establish what the selected historical releases published and whether those artifacts permit exact
retrospective reconstruction. They cannot establish a new speaker's intended meaning, replace domain-expert review for
novel ontology acquisition, or grant authority to register, mutate, call a service, act, or execute.
