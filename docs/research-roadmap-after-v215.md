# Research roadmap after V215

## Frozen conclusion

V215 selected `BOUNDED_EXTERNAL_PAYLOAD_DESIGN_ELIGIBLE`. Official metadata supports one carefully role-separated
successor: a bounded retrospective population derived from Uberon/OBO artifacts, accessed with version-aware
infrastructure and checked with W3C OWL conformance controls.

This advances the external-resource branch without changing its epistemic limit. Curated historical artifacts can
ground reconstruction of their own published semantics. They cannot establish the intended meaning of a new speaker
or replace expert adjudication for a new domain concept.

## Primary next track: V216 bounded payload protocol and population feasibility

V216 should prospectively freeze a minimal payload-bearing study before downloading any ontology or test repository.
Its design should specify:

1. exact Uberon artifact and historical-release identifiers, official URLs, retrieval dates, and expected media types;
2. per-artifact license and attribution preservation;
3. raw-file hashing before parsing or normalization;
4. a parser/reasoner validation set derived from a bounded, separately identified W3C OWL control payload;
5. provenance roles for source text, annotations, asserted axioms, imported axioms, inferred consequences, release
   metadata, and any derived benchmark label;
6. inclusion/exclusion rules for concepts, obsolete entities, imports, missing definitions, annotation-only changes,
   logical changes, merges, splits, and identifier replacement;
7. grouping that keeps synonyms, identifiers, descendants, mappings, and versions of the same concept together;
8. a development/protected split made before fitting reconstruction methods;
9. exact population and identifiability diagnostics before any language model is considered; and
10. negative stop rules for inadequate history, text/logic pairing, scale, parser validity, or license coverage.

The acquisition budget should be deliberately small: enough releases and concepts to test feasibility, not a bulk
mirror. V216 may retrieve only the exact payloads authorized by its own audited lock.

## Required scientific decomposition

The next population should distinguish at least four tasks rather than collapsing them into top-1 classification:

- **artifact recovery:** identify the historical class or axiom set already present in a frozen release;
- **version-change reconstruction:** infer which logical or definitional change occurred between two frozen releases;
- **equivalence and boundary diagnosis:** preserve multiple compatible classes or show a separating witness;
- **decision consequence:** quantify whether an incorrect merge, split, or forced interpretation changes a downstream
  shadow decision.

The first three can be grounded in published artifacts. The fourth should reuse the existing reversible sandbox and
certificate-aware planner only after the semantic population is independently valid.

## Deterministic-first sequence

After a valid population exists, the method sequence should remain:

1. exact parsing and normalization;
2. graph and identifier retrieval controls;
3. OWL reasoning and logical equivalence collapse;
4. version-diff and rule-based reconstruction;
5. candidate-set and residual analysis at fixed budgets;
6. a bounded local LLM candidate generator only if a meaningful, decision-relevant residual survives.

Any model should propose provisional candidates, not assign authority. API use is unnecessary unless a later,
separately locked comparison asks whether a stronger external model adds oracle-class recall beyond deterministic and
local-model controls.

## Deferred and excluded branches

- OAEI Anatomy remains deferred until a new prospective metadata step resolves payload licensing; V215's frozen pages
  cannot support it.
- OLS remains infrastructure only and must not be scored as independent semantic truth.
- W3C tests remain validation controls and must not be presented as open-world natural-language evidence.
- V213 protected data remain unopened.
- Strong B2c claims about correct novel speaker intent remain deferred pending appropriate speaker or domain-expert
  evidence.
- Registration, trusted-state mutation, service/tool action, and execution remain outside scope.

