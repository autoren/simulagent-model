# V97 Aggregate-Pool Open-Set Source Plan

## Material change from V96

V96 showed that individual SGD shards are too service-concentrated to receive whole benchmark roles.
V97 prospectively pins every remaining untouched development shard, `dev/dialogues_009.json` through
`dev/dialogues_020.json`, as one aggregate source pool. The complete 12-shard membership, immutable Git
blob identities, byte sizes, and selection rules are locked before any payload is opened.

Roles are assigned only after pooling source annotations:

1. sufficiently supported, previously unexposed services are identified from intent-activation counts;
2. exactly one hash-selected service is withheld completely as unsupported;
3. exactly three different services are hash-selected for the catalog;
4. within the catalog, one supported intent pair is hidden from each of two hash-selected services;
5. every other sufficiently supported catalog pair is declared known, with at least three remaining.

Role selection occurs before any lexical feature is computed. Services, rather than shards, are the
unit of the open-set partition.

## Retained class construction

Only source-annotated intent activations may supply known, novel-valid, or unsupported cases. Current
user-turn tokens alone determine familiar versus unfamiliar known requests. Genuine source `NONE`
states inside catalog services remain the only insufficient-evidence source. Previously exposed
services from V87–V96 are ineligible.

The unchanged noncompensatory class gates require at least 16 candidates in each of the five classes,
known coverage across at least two catalog services, novel coverage across exactly two hidden services,
insufficient-evidence coverage across at least two services, and unsupported coverage from exactly one
fully withheld service.

## Source-stage boundary

The one-shot census may automatically tokenize current turns only to count schema overlap. It may not
emit language, tokens, slot values, histories, or prompts. Passing authorizes only preregistration of a
dialogue-disjoint calibration/evaluation population. It does not authorize language extraction, manual
inspection, local or API model access, training, posterior integration, planning, or execution.
