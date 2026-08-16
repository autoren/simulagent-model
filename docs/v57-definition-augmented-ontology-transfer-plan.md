# V57 preregistration: definition-augmented ontology transfer

V56 verified the bounded execution of all 48 frozen planning policies. The next language-stage claim is deliberately narrower: can an exact compiler use supplied definitions to ground symbols it has never seen before, while the surrounding evidence grammar remains familiar?

## Isolation and claim

V57 introduces opaque unary predicates, binary relations, and bound actions. Each is supplied through a typed schema containing a controlled natural-language definition, its argument roles, and its permitted surface forms. Opaque identifiers are random and carry no lexical hint. The compiler must emit the exact typed AST, including sign, canonical relation order, action actor/target order, and the already known outer operation.

This is the report's first language stage: **declared definition plus new concept**. It is not human-authored open language. Both the definition language and the evidence grammar are declared and auditable. V57 therefore cannot support a claim about arbitrary definitions, human paraphrases, joint language/concept novelty, inference, or planning.

## Population and controls

An independent generator creates 16 ontology packs, each with two new unary predicates, two relations, and two actions. Three definition template families vary whether signature, meaning, or example comes first without changing the controlled definition grammar. Core records cross sign, orientation, operation, and argument-order factors. Targets remain unavailable until the population is sealed.

The primary compiler is schema-conditioned. Three controls are non-negotiable: removing definitions must cause abstention; opaque names alone must cause abstention; and definitions shuffled among same-kind symbols must not accidentally recover the target AST. A known-ontology control ensures the extension does not regress V40 semantics.

Safety cases cover missing or contradictory definitions, duplicate lexemes, incomplete signatures, type errors, ambiguous roles, and unknown terms. Six implementation mutations must all be killed before candidate construction.

## Decision rule

All 15 gates are noncompensatory and exactly one sealed evaluation is permitted. A pass qualifies only controlled definition-conditioned transfer to new typed symbols. It may authorize collection for the separately preregistered human-authored known-ontology track, but not the joint new-concept/new-surface cell.
