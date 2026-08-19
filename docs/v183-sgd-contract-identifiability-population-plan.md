# V183 SGD contract-identifiability population plan

## Question

Can the frozen, semantically non-colliding V134 SGD source be converted into a record-disjoint development/protected benchmark whose labels never require guessing an unobserved service version?

V183 is deliberately text-free and model-free. It reconstructs only source schemas and structured semantic frames for the already frozen 240 source identifiers. It does not emit or inspect utterances, dialogue text, slot values, or character spans.

## Capability identity

The unit of identity is not an intent name or a coarse domain composite. Each source service-intent definition is mapped to a semantic capability contract containing the domain, normalized service and intent description hashes, normalized intent name, transactionality, required/optional/result slot sets, and the complete slot type/value-description signature. Service version labels are excluded from the contract hash, so exact semantic duplicates cannot become distinct labels merely because SGD numbers them differently.

The six frozen V134 definitions remain declared known contracts. Valid-undeclared and unsupported source definitions remain shadow provisional and unsupported contracts respectively. A contract mapped from more than one truth kind is a benchmark confound, not a forced class label.

## Hidden identifiability screen

For each selected source record, V183 uses the source annotation only as a hidden screening oracle. The screen sees the normalized active-intent name and current non-value slot names. It does not see the service, domain, source-definition identity, truth kind, utterance, dialogue text, slot values, or spans.

A capability contract is compatible when its normalized intent name matches and it contains every observed slot name. The exact target contract must be retained. A singleton compatible set is `IDENTIFIABLE`; multiple compatible contracts, cross-kind collisions, and missing observations are `INSUFFICIENT`; an empty set is an invalid source record.

This is a conservative source-supervised upper-bound protocol. Passing it does not prove that a human or model can recover the semantic frame from the utterance. It only prevents a later language benchmark from assigning a unique answer where even the source's own declared semantic frame leaves multiple capability contracts possible.

## Frozen role split

Every one of the 66 V134 truth-by-presented-candidate cells contains four fixtures. A fixed hash order assigns exactly two to development and two to protected use. The split cannot depend on language, compatibility outcomes, or model scores. Each role must contain 132 fixtures: 120 source records and 12 missing controls.

## Noncompensatory gates

The formal build must reconstruct every selected identifier, retain every target contract, preserve exact role balance and disjointness, label every missing control insufficient, produce no invalid record or cross-kind contract collision, and retain a meaningful identifiable population in both roles. Each role must contain at least 48 identifiable source records, including at least 24 known, 12 provisional, and 6 unsupported records.

All persisted artifacts must remain text-free and value-free. Model/API access, manual language inspection, training, ontology registration, trusted-state mutation, service or sensor calls, side effects, action, and execution are all zero.

## Decision boundary

Passing authorizes only a separately preregistered, role-isolated language extraction. It does not authorize opening protected language, scoring a deterministic policy, running a local model or API, registering a capability, updating trusted state, acting, or executing.
