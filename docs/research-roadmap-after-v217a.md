# Research roadmap after V217A

## Frozen conclusion

V217A is a verified positive metadata-feasibility result. Of the three prospectively frozen independent sources, only
Mondo satisfied every noncompensatory source, event-diversity, lifecycle, ambiguity, bounded-asset, provenance, and
selection gate. Gene Ontology lacked two exact historical releases and change summaries in the frozen metadata. Cell
Ontology had rich bounded release evidence but no documented lifecycle-or-mapping change event in its five frozen
release summaries.

The selected Mondo pair is `v2026-07-06` to `v2026-08-04`, using `mondo-base.obo` assets of 49,322,038 and
49,407,821 bytes. The 98,729,859-byte pair satisfies the frozen V217A payload bound. V217A did not download either
asset.

## Scientific update

V217A identifies a better external-artifact target than the V216 Uberon pair for the intended representational
diagnosis:

- published additions, label changes, and definition changes provide observable semantic change;
- obsoletions with replacements and merged replacements provide lifecycle structure;
- stable identifiers and exact release assets permit reconstruction;
- merge and replacement histories create plausible non-singleton interpretations and separating witnesses; and
- exact release notes expose event families directly, so global ontology size is unnecessary as a proxy.

This remains an artifact-grounded result. Historical curator decisions can validate reconstruction of published Mondo
semantics, but they cannot establish what a new speaker meant by a new utterance.

## Primary next track: V218 prospective Mondo payload protocol

The next goal should design and audit one bounded payload study before downloading any selected asset. It should freeze:

1. the exact two `mondo-base.obo` asset URLs, byte counts, and GitHub-declared SHA-256 digests;
2. the exact current-release diff, obsoletion-candidate, and source-version control assets needed to check reconstruction;
3. a total download budget that includes every payload and forbids link expansion or imports;
4. raw-file hash verification before parsing;
5. a streaming OBO parser policy with no remote import resolution;
6. explicit provenance roles for old release, new release, release summary, diff control, source mapping, asserted text,
   asserted logical axioms, lifecycle annotations, and derived labels;
7. exact event definitions for addition, label change, definition change, asserted logical change, obsoletion,
   replacement, merge, and candidate-obsoletion status;
8. canonical concept-family grouping across identifiers, replacements, merged targets, synonyms, and versions;
9. development/protected assignment by concept family before any method fitting;
10. direct population gates for event diversity, usable text-plus-semantics, lifecycle examples, observational ambiguity,
    separating witnesses, group-disjoint scale, parser validity, and control agreement; and
11. a negative stop rule that freezes failure without payload, threshold, parser, source, URL, or protected-set tuning.

The design should not inherit V216's 20,000-term gate. It should gate the number and diversity of eligible concept
families and decision-relevant ambiguity directly.

## Conditional method track after V218

Only a verified positive V218 population may proceed to deterministic controls. The sequence remains:

1. exact parsing and version diff;
2. identifier, synonym, replacement, and graph retrieval;
3. asserted logical normalization and equivalence collapse;
4. contradiction and residual version-space handling;
5. exact oracle-class recall at fixed candidate budgets;
6. decision-consequence analysis in the existing reversible sandbox;
7. a bounded local LLM candidate generator only if a meaningful residual remains.

The model is optional. A valid external population does not by itself justify model use. Any later model must add
oracle-class recall beyond deterministic controls and must not turn a provisional historical candidate into authority.

## Secondary research tracks

The larger program remains divided into four tracks:

- **A — decision framework:** preserve certificate-aware abstention, reversible actions, and exact regret or consequence
  analysis.
- **B1 — artifact-grounded semantics:** reconstruct meanings and version changes that are answerable from frozen
  published artifacts. V218 belongs here.
- **B2c — new-speaker ontology acquisition:** remain deferred for claims about correct human intent unless appropriate
  speaker or domain-expert evidence becomes available. Existing ontologies can narrow candidates but cannot close this
  epistemic gap.
- **C — model contribution:** test local or API models only on a frozen residual as candidate generators under fixed
  budgets. No model is a semantic oracle.

Progress on B1 can continue without human review because the historical artifacts themselves provide the target.
Progress on B2c can continue only on weaker, explicitly scoped claims such as candidate coverage, safe abstention, or
synthetic recovery—not correct new-speaker intent.

## Preserved boundaries

- V216 and V213 protected data remain sealed.
- The Cell Ontology lifecycle-event gate is not relaxed, and the Gene Ontology URL set is not expanded post hoc.
- V217A's failed Mondo documentation URL remains a frozen failed attempt.
- No Mondo payload has yet been downloaded.
- No remote import, bulk mirror, ontology registration, trusted-state mutation, service/tool action, or execution is
  authorized.
- No local or API model is authorized before a separately frozen deterministic residual.
- Mondo evidence supports retrospective artifact reconstruction, not new-speaker intent or expert validation.
