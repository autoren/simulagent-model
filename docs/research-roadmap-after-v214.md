# Research roadmap after V214

## Frozen conclusion

V214 closed the typed program branch with `DETERMINISTIC_CLOSURE_ZERO_MODEL_ELIGIBILITY`. Retrieval and bounded
`L0/L+` synthesis were meaningfully imperfect, but complete 256-class constraint propagation and the unchanged
deterministic stack recovered every evaluation version space and shadow action with zero regret.

This means the project should not run an LLM on the current representation. The remaining scientific gap is upstream:
obtaining semantically grounded public language or definitions whose relation to executable behavior is not already
handed to the system as a typed evidence program.

## Primary next track: R1 external semantic-resource feasibility

The next stage should be a prospectively frozen metadata-first census of existing resources. Candidate sources should
be assessed without opening bulk ontology payloads during design. The census should seek:

- stable version identifiers and downloadable historical releases;
- clear redistribution and derivative-work licenses;
- textual labels, definitions, synonyms, examples, or competency questions;
- machine-readable logical axioms or other executable semantics;
- explicit mappings, alignment gold standards, or version-to-version changes;
- provenance sufficient to separate author text, curator decisions, inferred relations, and benchmark labels;
- enough independent concepts and change events for group-disjoint development/protected splits;
- an official or archival access path that can be content-hashed under a later lock.

Promising families include OBO Foundry ontologies exposed through EMBL-EBI OLS, OAEI alignment tracks with reference
alignments, and W3C OWL conformance/test resources. The census must verify current official metadata before selecting a
payload-bearing successor.

## What external resources can and cannot establish

Suitable curated resources can support a retrospective question:

> Given the text, axioms, mappings, and version history available to curators, can a system reconstruct the frozen
> equivalence class, detect a missing representation, preserve ambiguity, or identify a boundary witness?

They cannot establish that a proposed concept is the intended meaning of a genuinely new speaker. Historical curator
decisions are evidence about those historical artifacts, not a substitute for future speaker or expert adjudication.

## Conditional payload stage

If the metadata census finds at least one viable source, a separate lock may authorize a bounded payload retrieval and
population build. That stage should:

1. content-hash exact source releases before transformation;
2. preserve licenses and provenance per record;
3. derive behavioral or logical equivalence classes independently of surface labels;
4. group synonyms, mappings, and version variants before splitting;
5. separate development and protected payloads before method fitting;
6. exhaust deterministic parsers, reasoners, retrieval, and alignment controls;
7. authorize a local LLM only for a meaningful residual at a fixed candidate budget.

## Why not another synthetic language projection now

Earlier controlled-language branches and V214 already show that fully specified synthetic interfaces tend to collapse
under deterministic parsing or finite constraint propagation. Another hand-authored paraphrase layer would mostly test
whether the project remembered its own grammar. External curated definitions and version histories offer a more useful
next step while retaining executable reference structure.

## Other tracks

- The V213 protected typed partition remains unopened for downstream evaluation; it can be preserved as a future
  implementation-regression confirmation set, not used to manufacture model eligibility.
- The reversible sandbox and certificate-aware planner remain available if a later external or language branch produces
  provisional candidates. No execution machinery should be rebuilt.
- Strong B2c ontology-acquisition claims remain deferred until valid speaker or domain-expert evidence is available.
