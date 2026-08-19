# V217A Independent Source Event Metadata Census Results

## Outcome

V217A passed its prospectively frozen census and selected exactly one source:

```text
FRESH_INDEPENDENT_SOURCE_PAYLOAD_DESIGN_ELIGIBLE
selected source: MONDO
```

This is a metadata-feasibility result. It authorizes one separately locked Mondo payload-design protocol. It does not authorize an ontology download, protected-data access, model run, registration, trusted-state mutation, service action, or execution.

## What was frozen before source-specific inspection

The census fixed three independent source families and nine official metadata URLs:

1. Mondo;
2. Gene Ontology;
3. Cell Ontology.

It also fixed thirteen noncompensatory source dimensions, six semantic event categories, five ambiguity indicators, a maximum of 50,000,000 bytes per release asset, a maximum of 100,000,000 bytes for the smallest valid two-release pair, and lowest-priority-number selection among eligible sources. Total ontology term count was explicitly excluded as an eligibility criterion.

## Retrieval integrity

- Frozen URL attempts: `9 / 9`
- Successful metadata snapshots: `8`
- Frozen failure: `1` (`https://mondo.monarchinitiative.org/pages/releases/`, HTTP 404)
- URL accounting rate: `1.0`
- Successful snapshot hash coverage: `1.0`
- Assessment-dimension coverage: `1.0`
- True-claim evidence coverage: `1.0`
- Unexpected URL attempts: `0`
- Candidate payload downloads: `0`
- V216 protected accesses: `0`
- V213 protected accesses: `0`
- Model loads, generations, and API calls: `0`
- Training, registration, mutation, service action, and execution counts: `0`

The failed Mondo documentation URL was retained as a frozen negative retrieval record. No positive claim depends on it.

## Source decisions

### Mondo — eligible and selected

The frozen OBO Foundry record states CC BY 4.0 licensing, machine-readable release products, mapping provenance, and provenance-tracking intent. The frozen GitHub release metadata provides exact releases and byte-counted assets together with published changes for:

- additions;
- labels or renamings;
- text definitions;
- obsoletions and replacements;
- merged replacements.

The last two supply genuine lifecycle and ambiguity structure: a prior term may be merged into or replaced by another term, and some obsoletion candidates can later leave candidate status. These are control-relevant representational events rather than mere ontology size.

The bounded pair selected for a later design is:

| Release | Candidate asset | Bytes |
|---|---|---:|
| `v2026-08-04` | `mondo-base.obo` | 49,407,821 |
| `v2026-07-06` | `mondo-base.obo` | 49,322,038 |
| **Total** | | **98,729,859** |

Both per-asset and two-release byte gates pass. No asset was downloaded in V217A.

### Gene Ontology — not eligible under the frozen evidence

The frozen official pages establish authority, CC BY 4.0 licensing, textual content, and OBO/OWL/JSON products. The frozen GitHub releases query returned an empty array, however, and the prospectively frozen pages did not themselves identify two exact bounded historical releases with published change summaries. A link to a separate archives page was visible but was not followed because it was outside the frozen URL set.

Current availability was therefore not promoted into retrospective reconstruction evidence.

### Cell Ontology — rich but fails one event gate

Cell Ontology provides exact releases, bounded assets, CC BY 4.0 licensing, and detailed changes covering additions, labels, synonyms, text definitions, and logical relationships. It also publishes a stable SSSOM mapping product and cross-ontology link counts.

However, the five frozen release summaries did not document a mapping-change, merge, split, or term-obsoletion event. Static mapping availability is not a change event. Cell Ontology therefore failed the preregistered lifecycle-or-mapping event requirement even though its other source dimensions passed.

## Why Mondo passed when Uberon V216 did not

V216 failed because it used a frozen 20,000-term-per-release gate that the otherwise useful Uberon pair did not meet. V217A did not tune that threshold or reuse V216 protected records. It prospectively replaced term count with properties that better match the scientific question:

```text
exact releases
  + diverse semantic changes
  + lifecycle or mapping events
  + plausible observational ambiguity
  + bounded reconstruction
```

Mondo passed because its metadata shows all five properties. This does not retroactively change the V216 negative result.

## Claim boundary

V217A establishes only that one independent retrospective artifact source is suitable for a separately designed payload study.

It does not establish:

- that the payload parses or reconstructs as expected;
- that its change events yield useful exact equivalence classes;
- that a deterministic residual survives;
- that an LLM is needed;
- that historical ontology curators represent a new speaker;
- that any candidate meaning is correct for a new utterance;
- authority to register, mutate, call a service, act, or execute.

The historical artifact can supply external semantic supervision for an artifact-grounded study. It cannot substitute for speaker review in claims about a new person's intended meaning.

## Frozen artifact hashes before outcome verification

- Design: `e66b14165831b4cb54df2a50810b75a4909ea0ab704ec3326e4ee989c305e846`
- Design lock: `bd92bc5b2993cae27a768d3c12b62d78ff28a4a87d29bbdda5992745bc8e35b1`
- Retrieval manifest: `52608e70acc931aaa8a7b433dfc6525e0a3e47658e995863c73433643e1454ac`
- Evidence: `a763b1064cb8ed11c3dc27c8b4220f80bb29ba78e799ed2e2026dd206c6f3db6`
- Summary: `a3d65e31d776f053d5c40c513582c5b8f77b630688b2625f1af41be708e12866`
- Result: `8f23073f3d1531b30d3899f639efcf129d85d9d99afcc02db56329fba7ca46b6`

## Correct successor

The next experiment must be a new, separately audited Mondo payload protocol. Before any download it must freeze exact asset URLs and hashes available from metadata, parser and import policy, bounded extraction rules, release-event grouping, dev/protected separation, representational-diagnosis outputs, deterministic controls, and negative stop conditions.

The next stage should ask whether the exact Mondo pair can produce a valid artifact-grounded population with lifecycle ambiguity and decision consequences. It should not run an LLM merely because V217A passed.
