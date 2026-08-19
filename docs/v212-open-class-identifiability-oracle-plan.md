# V212 model-free representational-diagnosis oracle plan

## Revision status

This plan supersedes the unexecuted `{A, B, OUTSIDE_UNKNOWN}` observation-set scaffold. No V212 design audit, lock,
test execution, oracle result, or outcome exists. The old scaffold is marked `SUPERSEDED_BEFORE_LOCK_DO_NOT_AUDIT_OR_RUN`
and must be replaced in code and configuration before preregistration.

## Purpose

V211/V211r1 showed that a fixed calibration lexicon resolved every held-out residual record. V212 therefore moves
below language and asks which *behavioral meanings* are identifiable, how they relate to a frozen representation
language, and which decision is justified by the surviving evidence.

This is a development-only mechanism oracle over finite executable semantics. It is not a natural-language benchmark,
does not claim real-world ontology acquisition, and does not authorize a language model or ontology mutation.

## Orthogonal diagnoses

V212 freezes a base DSL `L0`, a diagnostic extension language `L+`, and a finite complete behavioral domain. Every
syntactic hypothesis is evaluated on the complete domain and collapsed into its behavioral equivalence class.

Each concept episode receives two independent diagnoses.

### Expressibility relative to the frozen languages

- `EXISTING_PRIMITIVE`: equivalent to one registered primitive in `L0`;
- `EXISTING_COMPOSITION`: expressible in `L0` but not equivalent to a registered primitive;
- `MISSING_OPERATOR`: not expressible in `L0` but expressible in `L+` through a prospectively withheld operator; or
- `IRREDUCIBLE_PROVISIONAL`: not expressible in either frozen language and retained only as an executable shadow
  behavior, not proof of an absolute new primitive.

### Evidence status

- `SUFFICIENT`: authorized evidence leaves one behavioral equivalence class;
- `AMBIGUOUS`: at least two distinguishable equivalence classes survive; or
- `CONTRADICTORY`: no admissible equivalence class survives.

An outside or representation-gap diagnosis never implies sufficient evidence. Conversely, ambiguity never implies a
new primitive. Candidate programs, representational diagnoses, and evidence statuses remain separate artifacts.

## Frozen concept families

The oracle must include balanced episodes for:

- exact aliases of existing primitives;
- nontrivial existing compositions;
- near aliases that differ on at least one exact boundary input;
- broader and narrower extensions;
- meanings requiring a withheld operator;
- behaviors irreducible relative to `L0` and `L+`;
- arbitrary renamed symbols grounded only by explicit in-episode reference evidence;
- intentionally ambiguous evidence; and
- contradictory evidence.

The unit is a concept episode, not a paraphrase. Multiple definitions, renamings, or evidence-order variants are
counterfactual robustness views of the same semantic target and may not inflate the independent sample count.

## Exact candidate and witness contract

The oracle enumerates or symbolically represents every admissible hypothesis, filters it only with the authorized
evidence program, and returns the complete set of surviving behavioral equivalence classes. It must never select a
single syntax merely because equivalent alternatives remain.

For every pair of distinct surviving classes, complete enumeration must return a boundary witness on which their
behavior differs. For every declared equivalence, no witness may exist over the complete domain. Reference-grounded
symbols must become identifiable with their full evidence program and ambiguous when each prospectively essential
fact is ablated.

Bijective vocabulary renaming, evidence ordering, commutative expression ordering, and equivalent syntactic rewrites
must preserve candidate equivalence classes and justified decisions exactly.

## Non-authoritative decision contract

The diagnosis supports only shadow research decisions:

- reuse an existing primitive;
- construct an existing composition;
- diagnose a missing representation operator;
- retain an irreducible provisional behavior in a shadow namespace;
- request a distinguishing boundary observation; or
- defer/adjudicate a contradictory state.

No decision registers a concept or executes it. Forced primitive creation, forced nearest-concept merging,
closed-world collapse, and abstain-always are fixed comparators. False primitive creation and false merging are
separate noncompensatory gates, and comparator regret must be measured under the hidden executable semantics.

## Information firewall

The prediction process receives the public base language, public episode definition/evidence, and any explicitly
authorized reference facts. Hidden target behavior, expressibility diagnosis, evidence status, boundary oracle, group
identity, protected role, and decision truth remain sealed. Predictions are frozen by hash before any truth join.

V212 itself uses no natural-language surface population. Its public programs and hidden executable truths are generated
only after the replacement design is audited and locked.

## Successor sequence and stop rules

If every V212 integrity, equivalence, witness, diagnosis, decision, and access gate passes, it authorizes only V213
design: a fresh role-separated programmatic concept population with definitions, positive/negative/boundary evidence,
renamings, and hidden executable semantics.

V214 then applies deterministic retrieval, controlled parsing, bounded synthesis, exact version-space filtering,
equivalence collapse, boundary-witness generation, contradiction handling, and complete safe retention. A zero
residual closes the LLM branch successfully. A residual authorizes a local model only if it is truth-blind,
nontrivial, factor-complete, behaviorally identifiable, decision-relevant, and not caused by a missing deterministic
control.

Any later model receives a fixed candidate budget and proposal authority only. It must improve protected
oracle-equivalence-class recall beyond the strongest deterministic generator without making verification or
elicitation more expensive through uncontrolled candidate growth.

The previously verified sandbox and certificate-aware planner are reused only after real candidates survive V214 or
the conditional model study. They are integration dependencies, not rescue mechanisms for failed identifiability.

After the V212 schema is frozen, a separate metadata-first census may assess versioned ontologies and curated
alignment benchmarks for retrospective confirmation. Such data can establish reconstruction of independently curated
semantics and safe provisional staging without new participants; they cannot establish prospective discovery of
correct new speaker or domain meaning.

## Prohibited scope

V212 performs no natural-language reads, external ontology payload reads, protected access, model loading or
generation, API call, training, ontology registration, trusted-state mutation, service call, side effect, real action,
or execution. No V213 population, V214 baseline, external-resource extraction, or model run is authorized until its
own prospective lock.
