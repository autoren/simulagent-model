# V215 external semantic-resource metadata census plan

## Purpose and prior exposure

V214 closed the current typed program representation with zero model eligibility. V215 asks whether existing curated
resources can support a stronger retrospective semantic-reconstruction benchmark without requiring new human review.

OBO Foundry, EMBL-EBI OLS, OAEI, and W3C OWL resources were already known at a high level from earlier research. V215
is therefore not blind source discovery. Before the formal census, it freezes exact URLs, evidence fields, mandatory
criteria, scoring, and the payload-stage decision. No formal record has yet been scored and no bulk ontology, alignment,
or test-suite payload has been downloaded.

## Frozen source units

The census evaluates four role-distinct units:

1. `OBO_UBERON_VIA_OLS_COMPOSITE` — a candidate retrospective benchmark combining an open, versioned OBO ontology with
   OLS metadata/access infrastructure.
2. `OAEI_ANATOMY_REFERENCE_ALIGNMENT` — a candidate alignment benchmark with source ontologies and curated reference
   correspondences.
3. `W3C_OWL2_CONFORMANCE_TESTS` — a validation/control resource for logical reasoner correctness, not by itself a
   natural-language semantic benchmark.
4. `EMBL_EBI_OLS_INFRASTRUCTURE` — discovery, metadata, and API infrastructure, not by itself a source of independent
   semantic ground truth.

Only frozen official URLs may support the formal scores. A failed or missing URL is recorded as a negative; it cannot be
silently replaced after the lock. Search may be used only to locate the frozen official page when the exact redirect is
handled by the official domain.

## Evidence schema

Every unit records official URLs, retrieval status, final URL, title, retrieval date, SHA-256 of the stored metadata
snapshot, concise paraphrased evidence, and a boolean assessment for:

- official authority;
- stable retrieval;
- explicit versioning or archived releases;
- explicit license or governing open-use policy;
- textual semantic content;
- machine-readable logical semantics;
- curated reference mappings or version/change history;
- separable provenance roles;
- credible scale for group-disjoint study;
- support for a retrospective reconstruction task.

Each boolean must cite at least one frozen metadata snapshot. Missing or ambiguous evidence scores false. V215 does not
infer a license from general availability.

## Eligibility roles

A `PAYLOAD_BENCHMARK_CANDIDATE` must satisfy every mandatory dimension: authority, retrieval, versioning/archive,
license/open policy, textual content, logical semantics, mapping/change evidence, provenance, scale, and retrospective
task support.

A `VALIDATION_CONTROL` must establish authority, retrieval, licensing/use policy, machine-readable logical tests,
provenance, and retrospective validation utility; textual definitions or change histories are not mandatory.

An `INFRASTRUCTURE_ONLY` unit may be useful but cannot independently authorize a benchmark payload.

## Gates and decision

V215 passes only if:

- all four source units and all frozen URL attempts are accounted for;
- every successful metadata snapshot is stored and content-hashed;
- every assessment has source-linked evidence and no unsupported true value;
- at least one payload benchmark candidate is eligible;
- at least one validation/control resource is eligible;
- no selected payload candidate has unresolved license, provenance, version, or access status;
- the recommended payload design names exact source roles and preserves the “retrospective artifact, not new speaker
  intent” boundary;
- all protected/model/training/mutation/action access counts remain zero.

Passing authorizes only a separate payload-retrieval and population-design protocol. It does not authorize downloading
the ontology/alignment/test payload, running a model, registering a concept, or acting.

Failure freezes the missing dimensions and either selects a narrower source or stops the external-resource branch; it
does not relax the evidence requirements.
