# V145 finite registered certificate codebook plan

## Question

V144 confounded semantic evidence production with free-form thinking completion: 24 traces never closed. V145 asks a model-free architectural question that does not reinterpret those traces: can every certificate required by the frozen controlled task be represented as one of a small number of registered alternatives, so a future model can score alternatives rather than generate a certificate?

## Interface

The codebook contains:

- eight sufficient singleton codes, one for each non-`A00` choice;
- six insufficient codes, one for each registered ambiguity pair.

Selecting a code deterministically reconstructs its exact V142-compatible certificate and final choice. Unknown, malformed, or unavailable selections fail closed to programmatic `A00`.

A future realization would score all 14 fixed codes as closed alternatives. It would not generate reasoning or certificate JSON. Scores would remain non-authoritative and could not prune the safe hypothesis universe. Same-model scores or passes would not count as independent evidence.

## Audit

V145 constructs an abstract 288-row structural census matching the six-family, eight-group, six-stage V142 topology without reading any utterance. The oracle must map every row to an exact registered code and certificate. Invalid-code mutations must all fail closed. The audit must also demonstrate that a registered but semantically wrong singleton remains undetectable by structural validation.

Passing establishes only interface feasibility and authorizes design of a fresh population and a separately preregistered scoring protocol. It does not authorize a language or model run, the V142 test split, APIs, training, induction, authority, action, or execution.
