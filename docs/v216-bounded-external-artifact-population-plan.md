# V216 bounded external-artifact population plan

## Question

V215 established metadata feasibility. V216 asks the next, narrower question:

> Do two small, frozen Uberon releases contain enough paired textual and asserted-logical change evidence to build a
> role-separated retrospective reconstruction population, while preserving exact provenance and a protected split?

This is not a language-model experiment and not a new-concept acquisition study. It concerns only already-published
artifacts.

## Prior exposure and prospective boundary

Before this design lock, official release/archive metadata and response headers were inspected to choose exact files
and a safe byte budget. No ontology or W3C test payload body, term, definition, or axiom was read. The study therefore
does not claim blind source discovery, but payload contents and formal population records remain prospective.

The selected files are:

- `uberon-basic.obo` from Uberon `v2025-05-28`;
- `uberon-basic.obo` from Uberon `v2025-08-15`; and
- the W3C 18 November 2009 archived OWL 2 `all.rdf` test aggregation.

The two OBO files total about 23.8 MB. The W3C control is about 3.1 MB. The frozen total ceiling is 28 MB, with exactly
three listed requests and no substitutions.

## Roles

The older Uberon release supplies the prior published state. The newer release supplies the retrospective target.
Their CC BY 3.0 attribution and exact release identity remain attached to every derived record.

The W3C RDF/XML file is a parser-structure control only. A successful XML parse and subject census do not establish
OWL reasoning correctness. No inferred semantic-equivalence claim is made; V216 classes are hashes of normalized
asserted OBO axiom fields.

## Parsing and reconstruction population

The frozen parser reads `[Term]` stanzas, normalizes whitespace and set-valued fields, and separates:

- text: name, definition, synonyms;
- asserted logical structure: `is_a`, `relationship`, `intersection_of`, `equivalent_to`, and `disjoint_from`; and
- lifecycle status: obsoletion, replacement, and consideration links.

An eligible current term must have a current name, definition, at least one asserted logical axiom, and at least one
frozen change type. Removed and unchanged terms and terms missing required evidence are excluded without imputation.

Public observations contain prior/current names, definitions, and synonyms after source-identifier redaction. Hidden
truth contains source IDs, change labels, asserted axiom states/deltas, and the oracle class. Identical normalized
public observations define one version-space group and always remain in the same split. Their oracle candidate set is
the set of all asserted-axiom classes compatible with that observation; ambiguity is preserved rather than forced to a
single label.

## Split and integrity gates

Groups are stratified by primary change type and ranked by SHA-256; every fourth group is protected. At least 24
groups, two primary change types, 16 development groups, and eight protected groups must survive. Definitions and
asserted logical targets must have complete coverage. Version spaces and axiom-difference boundary witnesses must
reconstruct exactly, source IDs must not leak into public text, and cross-split group overlap must be zero.

Both Uberon files must contain at least 20,000 parsed terms. The W3C control must parse as RDF/XML and expose at least
100 RDF subjects. All exact byte counts, raw hashes, declared digests, and licenses are noncompensatory.

## Decision

A pass authorizes only V217 design: deterministic retrieval, graph, version-diff, and reconstruction controls on the
development partition. V216 does not authorize opening the protected partition for method evaluation or running a
local/API model.

A future local model becomes eligible only after a separate lock if deterministic controls leave at least 12 genuine
development residual groups and a fixed-budget model can be tested for incremental oracle-class recall. Any planner
claim additionally requires a decision-relevant residual. Top-1 accuracy alone is insufficient, and an API is not
required.

Failure freezes a negative feasibility result without changing releases, inclusions, split logic, thresholds, or
gates after seeing the payload.

