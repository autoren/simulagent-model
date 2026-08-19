# V199 exact menu-presentation robustness plan

## Question

Does the confirmed finite-menu interface admit an exact nuisance-shift test in which option order and identifiers
change but option semantics, target expressibility, and trusted-controller behavior do not?

V199 is a model-free feasibility and preregistration step. It does not read language or evaluate the model.

## Population and transformations

Use the 98 V191 development identities: 84 observed records, six per contract, and 14 missing controls. For each
identity construct two variants from fixed SHA-256 salts and the public `record_id` only:

1. `ORDER_ONLY`: permute presentation order while preserving each canonical `Mxx` identifier.
2. `ORDER_AND_OPAQUE_ID`: assign the 14 semantic contracts bijectively to `Q01` through `Q14`, then independently
   permute presentation order.

The transform receives no conversation, utterance, truth kind, model output, score, or error. If a hash ordering were
identical to canonical order, rotate it once so every record is a genuine order perturbation.

## Exact invariants

For every record and variant, independently verify:

- exactly 14 visible options and 14 hidden mappings;
- the exact canonical multiset of `(domain, intent_concept)` pairs;
- a one-to-one option-ID-to-contract map;
- the record's hidden target appears exactly once;
- `ORDER_ONLY` preserves the canonical ID-to-contract map;
- the opaque variant uses exactly `Q01` through `Q14`;
- presentation order differs from canonical;
- dynamic invalid-output parser controls fail closed; and
- the trusted `OTHER`, hierarchy fallback, 0.20 top-3 question cost, and complete candidate universe are unchanged.

Visible artifacts may contain only record identity, observation availability, variant identity, and the three menu
fields. Target, contract mapping, source, truth-kind, and language fields remain separate.

## Prospective paired model gates

V199 also freezes the later development gates before transformed generation. Each variant must lose no more than
0.05 primary or macro top-3 recall from V195, add no more than 0.02 primary cost, preserve at least 0.80 top-1
contract agreement and 0.80 mean top-3 contract-set Jaccard with canonical output, and disagree with canonical target
inclusion on no more than 5% of records. It must still improve by at least 0.01 over `CHAR_LAST` evaluated on the same
transformed menus. Structural validity must be at least 0.98; final truncation and false terminal decisions must be
zero; target retention and trusted exactness must remain one.

These are robustness gates, not a promise that the model will pass them. A miss is frozen without tuning.

## Access and stop rules

V199 reads only frozen identities, hidden targets, and canonical menu metadata. Utterance reads, deterministic
language scores, model/API/training access, protected language, ontology mutation, services, side effects, action, and
execution are zero. A pass authorizes only a separately locked deterministic development evaluation.

