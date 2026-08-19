# V96 Two-Source Activation Open-Set Source Plan

## Material change from V95

V95 validated the activation-turn catalog construction but had only three eligible fresh services, so
it could not both preserve a three-service catalog and withhold a fourth service as unsupported. V96
does not relax that requirement or reuse V95 language. It prospectively assigns two untouched official
SGD shards to disjoint roles:

- `dev/dialogues_007.json` supplies exactly three hash-selected catalog services;
- `dev/dialogues_008.json` supplies exactly one hash-selected service that is absent from the catalog
  and completely withheld as unsupported.

Both immutable source identities and roles are locked before either payload is downloaded.

## Retained construction

Only source-annotated intent activations may enter known, novel-valid, or unsupported classes. Current
user-turn tokens alone determine familiar versus unfamiliar known requests. Within the catalog, two
different services are hash-selected and one sufficiently supported intent pair from each is hidden.
At least three supported intent pairs must remain declared. Genuine source `NONE` states inside the
catalog remain the only insufficient-evidence source.

Service and pair selection uses source annotations and deterministic hashes before lexical overlap is
computed. Catalog and unsupported services must be disjoint and previously exposed services are
ineligible for both roles.

## Source-stage boundary

The one-shot inventory may automatically tokenize current turns only to count schema overlap. It may
not emit language, tokens, slot values, histories, or prompts. Passing authorizes only a later,
dialogue-disjoint population and calibration/evaluation preregistration. It does not authorize language
extraction, manual inspection, model access, API use, training, posterior integration, planning, or
execution.
