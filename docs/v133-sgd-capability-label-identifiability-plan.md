# V133 SGD Capability-label Identifiability Audit Plan

## Question

V132 used SGD's structural open-set label: a test service-intent pair absent from train but in a train-seen
domain is “novel valid.” That can represent a new capability, but it can also represent a new service version
that reuses a declared capability. V133 asks whether the selected V131 novel labels are distinguishable
capabilities under the source schemas.

## Frozen model-free audit

Read only the pinned train and test schema files. For every selected novel service-intent definition, compare
it with all six declared known definitions using four exact, deterministic relations:

1. Unicode-normalized, case-folded alphanumeric intent-name equality;
2. normalized description equality;
3. exact required- and optional-slot set equality;
4. equality of name, description, and slot signature jointly.

Weight pair collisions by the 72 frozen novel fixtures and report choice-level coverage. Emit identifiers,
hashes, and collision booleans only—no raw descriptions, utterances, slot values, model responses, or manual
semantic judgments.

## Gates

All three novel composites must have no exact name collision with a declared known choice. At most 10% of
selected novel records may collide by exact name or full signature, and no novel composite may consist
entirely of name-colliding definitions. Every definition must resolve exactly once.

Failure means V132 remains a valid negative for its frozen service-schema classification, but it cannot be
interpreted as a pure test of novel-capability understanding. Failure authorizes only a new text-free source
design built from semantically non-colliding capability labels—not a model rerun, prompt repair, scale-up,
induction, API, training, authority, or execution.
