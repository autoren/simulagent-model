# V215 external semantic-resource metadata census results

## Bottom line

V215 passed every frozen integrity, census, role-separation, claim-boundary, and access gate. It selected branch
`BOUNDED_EXTERNAL_PAYLOAD_DESIGN_ELIGIBLE`.

The result authorizes a separate bounded payload-retrieval and population-design protocol around an
OBO/Uberon/OLS composite, with W3C OWL conformance resources used only as a logical validation control. It does not
authorize a payload download, model run, ontology registration, trusted-state mutation, action, or execution.

## Frozen retrieval outcome

The census made exactly 12 preregistered `(source unit, URL)` attempts across four role-distinct units. Eleven
succeeded and every successful response was stored and SHA-256 hashed. One URL—the preregistered OBO versioning
principle path—returned 404 and remained a recorded failure; it was not replaced after the lock.

| Source unit | Intended role | Successful URLs | True dimensions | Eligible? | Main limitation |
|---|---|---:|---:|---|---|
| OBO/Uberon via OLS composite | Payload candidate | 3/4 | 10/10 | Yes | Versioning page failed, but successful OLS and Uberon metadata still established version-aware access and maintained products |
| OAEI Anatomy reference alignment | Payload candidate | 3/3 | 9/10 | No | No explicit license or governing open-use policy on the frozen pages |
| W3C OWL 2 conformance/tests | Validation control | 2/2 | 9/10 | Yes | Frozen metadata did not establish group-disjoint population scale; this was not mandatory for the control role |
| EMBL-EBI OLS | Infrastructure only | 3/3 | 5/10 | Yes for infrastructure | Client-shell/API metadata is not independent semantic ground truth |

URL accounting, snapshot-hash coverage, assessment coverage, and evidence coverage were all `1.0`. There were no
unexpected URL attempts and no unresolved mandatory dimensions for the selected payload candidate.

## Why the selected roles are defensible

The official Uberon record supplies a stable PURL, named OWL/OBO products, a human-readable cross-species anatomy
scope, source repository and tracker, dependencies, contact information, and an explicit CC BY 3.0 license. The OBO
open principle separately states the Foundry's licensing requirements. The OLS metadata describes version-aware
access to biomedical ontologies. Together, these establish metadata feasibility for a retrospective artifact study.

The W3C Recommendation defines exact syntactic and semantic conformance conditions, machine-processable OWL test
formats, premise/conclusion roles, and expected entailment or consistency outcomes. It is therefore an appropriate
logical-processing control, not a natural-language benchmark.

OLS exposes ontology, term, property, and individual endpoints and can serve as retrieval infrastructure. Its outputs
remain attributable to their ontology providers; OLS itself is not treated as the truth source.

OAEI is scientifically attractive because its annual Anatomy track distinguishes source ontologies, reference
mappings, organizers, dataset improvers, evaluated systems, and results. V215 nevertheless excludes it from the next
payload stage because the locked pages do not state a license. General availability was not allowed to stand in for
reuse permission.

## Claim boundary

This is a metadata-feasibility result only. It does not show that:

- Uberon records will yield a useful benchmark population after exact payload inspection;
- text uniquely determines historical logical axioms or curator choices;
- a model understands open-world language;
- a reconstructed concept matches a new speaker's intended meaning;
- expert review can be simulated; or
- any proposed concept may be registered, trusted, or executed.

The permitted successor is a new prospective protocol that freezes exact payload identifiers, versions, hashes,
licenses, extraction rules, provenance roles, grouping, splitting, and stop conditions before bounded retrieval.

