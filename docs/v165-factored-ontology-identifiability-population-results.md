# V165 factored ontology-identifiability population results

## Outcome

V165 passed every prospectively frozen population, factorization, identifiability, projection, renaming, access,
and authority gate. It creates the model-free development substrate required after V164 without loading a model
or claiming absolute novelty.

The population contains 144 project-authored synthetic development records in a complete 3×3 factorial:

| Generative expressibility | Sufficient | Ambiguous | Contradictory | Total |
|---|---:|---:|---:|---:|
| Alias of a registered primitive | 16 | 16 | 16 | 48 |
| Registered bounded composition | 16 | 16 | 16 | 48 |
| Provisional primitive relative to the DSL | 16 | 16 | 16 | 48 |
| Total | 48 | 48 | 48 | 144 |

Each cell has four logical target functions under four independent entity/primitive renamings. There are 36
logical target groups and four namespaces.

## Exact finite-DSL proof

The frozen domain has three Boolean unary predicates and all eight possible valuations. Exact enumeration covers
all 256 Boolean truth tables. Nine are registered in the bounded DSL: three aliases and six distinct binary
AND/OR compositions. The other 247 are called provisional only because they are not extensionally equivalent to
one of those nine functions on the exhaustive domain.

The exact parser and version-space enumerator proved the intended evidence contracts for every record:

- all 48 sufficient records retained exactly one candidate;
- all 48 ambiguous records retained exactly 64 candidates spanning alias, composition, and provisional classes;
- all 48 contradictory records retained zero candidates because the same intervention had opposing labels;
- the generative target was retained in 100% of noncontradictory records;
- sufficient expressibility classification and evidence-status classification were both 100%; and
- version-space size and expressibility-class coverage were invariant under all renamings.

This is a representation result, not a classifier result. In ambiguous records the system correctly refuses to
recover the hidden generative class: multiple extensionally different functions from all three classes remain
consistent. In contradictory records it retains none rather than inventing a repair.

## Public/private boundary

Public records contain only the frozen ontology names, controlled definition, and typed positive/negative
intervention observations. Target truth tables, generative factors, version spaces, parser outcomes, and
identifiability contracts remain private. Public projection was exact and hidden-field leakage was zero.

There is no evaluation population. Manual judgments, model loads or generations, API calls, training, ontology
registration, services, side effects, and execution were zero. The authoritative ontology remained immutable;
all 256 candidates were shadow-only.

## Decision

Freeze V165:

`freeze_V165_and_authorize_separate_model_free_deterministic_baseline_preregistration_only`

Passing authorizes only V166's prospective model-free comparison of controlled definition parsing, ontology
retrieval, exact version-space enumeration, safe complete retention, and oracle controls. It does not authorize
immediate scoring, an evaluation population, an LLM, API, training, provisional registration, belief or action
authority, or execution.

## Claim boundary

V165 is project-authored finite-DSL development evidence with automatically checkable ground truth. It shows that
expressibility relative to a bounded language can be represented separately from evidence sufficiency. It is not
external language evidence, unrestricted ontology learning, human validation, absolute novelty, deployment
safety, action, or execution.
