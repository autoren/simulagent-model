# Research roadmap after V223

## Frozen conclusion

V223 establishes that Mondo's documented structured term-request workflow is eligible for a source-specific B2c
acquisition study. It is the only one of four frozen workflow families to pass all 17 mandatory metadata and
provenance dimensions. This does not yet establish that individual records provide clean independent gold, balanced
outcome strata, or enough language-level evidence.

The claim is deliberately limited to **structured ontology-change requests**. It does not extend to unrestricted
open-world conversation.

## Track map

### Track A — retrospective ontology reconstruction: closed positive

V219A–V221/V221r1 showed that versioned Mondo artifacts and deterministic exact-family expansion suffice for the
frozen retrospective reconstruction task. There is no model-eligible residual, so model scaling remains closed on
this track.

### Track B2a/B2b — representational and decision mechanics: established

V212–V214 remain the reusable model-free sequence: diagnose exact representational states, create role-separated
concept populations, and exhaust deterministic candidate/version-space controls. Later language work should reuse
these controls rather than replacing them.

### Track B2c — external semantics: conditionally reopened for one source

V223 conditionally reopens B2c for Mondo's archived human term-request workflow. The evidence source is archived
requester and curator behavior, not simulated people, project-authored labels, or model confidence.

The next experiment is **V224: Mondo record/disposition metadata census**. It is an acquisition-feasibility audit,
not a language benchmark.

## V224 sequence

### Stage 1 — prospective design and firewall

Before requesting any record endpoint, freeze:

- exact Mondo repository revision and retrieval date;
- an immutable sampling frame or reproducible metadata query;
- a date window chosen without inspecting request language;
- permanent exclusion of every identifier exposed during discovery;
- exclusion rules for AI-assisted, bot-only, reopened, migrated, test, and ambiguous records;
- disjoint group keys and minimum counts by outcome;
- an endpoint allowlist restricted to structured metadata, labels, identities, timestamps, state transitions,
  cross-references, and release containment; and
- zero body, comment text, review text, or model access.

### Stage 2 — metadata-only disposition census

For each candidate record, use only metadata to test whether the intended outcome can be established without semantic
guessing:

1. `ACCEPTED_NEW`: independent human approval plus merged change, new stable identifier, and first containing release;
2. `EXISTING_OR_DUPLICATE`: explicit duplicate/existing disposition linked to the prior term;
3. `INSUFFICIENT_OR_CLARIFY`: explicit clarification/blocked state whose documented meaning is missing evidence;
4. `UNSUPPORTED_OR_OUT_OF_SCOPE`: explicit scope or unsupported disposition; and
5. `AMBIGUOUS_EXCLUDE`: conflicting, free-text-only, automated-only, reopened, or otherwise non-auditable history.

The census must prove requester/reviewer independence per retained record and report all exclusions. A label name alone
is not enough when its semantics or event author is unclear.

### Stage 3 — branch gate

Open no request language unless the metadata census proves prospectively fixed minimum counts in all four semantic
strata, exact provenance, group-disjoint roles, and sufficiently many independently adjudicated records.

- If the gate passes, design V225 as a separately locked, role-separated language acquisition and identifiability
  study. Request bodies would remain sealed until that design passes.
- If the gate fails, freeze B2c as externally unvalidated for this source and do not relax the outcome taxonomy by
  reading language.

### Stage 4 — only after a later language study

If a meaningful, decision-relevant deterministic residual survives exact retrieval, parsing, bounded synthesis,
equivalence collapse, and contradiction controls, a bounded local LLM may be tested only as a candidate generator.
Its endpoint is incremental oracle-class recall at a fixed candidate budget, not top-1 authority. API use is not
needed unless a later prospective study isolates a capacity question that local models cannot answer.

## Preserved safety and validity rules

- Archived human adjudication may replace new live review only when independence and semantics are demonstrable per
  record; it may not be imitated with synthetic people.
- Catalog absence, closure, model agreement, confidence, or an apparently plausible nearest term are never gold.
- Candidate concepts remain provisional and non-authoritative.
- No model may register a concept, mutate trusted state, authorize service action, or execute.
- No protected dataset is opened merely because an upstream feasibility gate passes.

