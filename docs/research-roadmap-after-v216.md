# Research roadmap after V216

## Frozen conclusion

V216 is a formally negative result. Its only failed scientific check was the preregistered minimum of 20,000 parsed
terms per Uberon release: the two locked `uberon-basic.obo` files contained 15,719 and 15,763 terms. V216r1 verified
and froze that negative outcome without changing the gate or authorizing V217.

The payload itself was intact, and the derived population was not small: 197 clean, group-disjoint records survived,
including 44 additions, 152 primary logical-axiom changes, and one primary definition change. Those observations are
descriptive boundary evidence, not grounds for overriding the failed gate.

## Scientific update

V216 changes source selection in two ways:

1. **Global ontology size is not the right feasibility proxy.** The relevant quantity is the number and diversity of
   independently usable version-change events with paired text and asserted semantics.
2. **Change diversity and ambiguity must be visible before payload selection.** This Uberon pair was dominated by
   asserted logical changes, had only one primary definition change, and produced no identical public observations
   with multiple oracle classes. It is useful for version-diff engineering but weak for the intended open-world
   language and abstention question.

The project must not lower V216's term threshold, reuse its protected partition, or relabel its 197 records as a passed
benchmark.

## Primary next track: independent external-source event census

The next stage should be metadata-only and independent of the opened Uberon pair. It should prospectively freeze a
small set of new OBO-compatible source families before reading their release payloads. Candidate families may include
Mondo, Gene Ontology, or Cell Ontology, but eligibility must be determined from official metadata rather than name or
reputation.

The census should look for source-visible evidence of:

- exact historical release identifiers and bounded assets;
- explicit license and attribution;
- published release diffs or change summaries;
- multiple kinds of text change, logical change, addition, obsoletion, replacement, merge, or split;
- stable identifiers and provenance sufficient to group versions and replacements;
- a plausible mechanism for genuine observational ambiguity or incomplete evidence; and
- enough independent change families for a future group-disjoint development/protected split.

Selection should use noncompensatory metadata criteria. It should not set a new total-term threshold calibrated from
the observed V216 counts. If no independent source exposes adequate event and provenance metadata, the external
retrospective branch should stop rather than open more payloads speculatively.

## Conditional successor after a positive census

Only a fresh positive metadata census may authorize another bounded payload protocol. That protocol should:

1. freeze exact files, byte budgets, media types, licenses, and hashes before retrieval;
2. define change-event and observation representations without using V216 protected records;
3. gate directly on eligible paired events and change-family diversity, using source-independent thresholds justified
   before payload exposure;
4. preserve ambiguous version spaces rather than require singleton labels;
5. validate asserted parsing separately from inferred logical equivalence; and
6. freeze a development/protected split before method fitting.

## Method and model sequence remains unchanged

If a valid fresh population is eventually obtained, deterministic parsing, graph retrieval, version-diff rules,
logical normalization, and candidate-set controls come first. A local model may be tested only on a separately frozen,
meaningful residual at a fixed candidate budget. It must add oracle-class recall rather than merely improve top-1
accuracy. An API model is optional and not required.

Any downstream planner claim still requires a decision-relevant residual and must reuse the existing sandbox and
certificate-aware architecture. Candidate proposals remain provisional and non-authoritative.

## Preserved boundaries

- The V216 protected partition is sealed from downstream method evaluation.
- V213 protected data remain unopened.
- OAEI remains excluded because V215 did not resolve licensing on its frozen pages.
- OLS remains access infrastructure, and W3C remains a parser/logical validation control.
- Historical curator artifacts do not establish the intended meaning of a new speaker.
- Strong B2c ontology-acquisition claims still require appropriate speaker or domain-expert evidence.
- Registration, trusted-state mutation, service/tool action, and execution remain outside scope.

