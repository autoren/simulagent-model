# V166 model-free factored ontology baseline results

## Outcome

V166 passed all preregistered gates. On the 144 project-authored development records, the frozen controlled
definition parser plus exact intervention filter reproduced every hidden version space and evidence status.
It retained every noncontradictory target, resolved every sufficient case, preserved every ambiguous case,
detected every contradiction, and was invariant to all four primitive renamings.

The 48 intentionally ambiguous records each retained exactly 64 candidates. They are correct unresolved
evidence states, not errors. The prospective model-eligible residual contained zero records. No model, API,
training run, manual judgment, ontology registration, service call, side effect, action, or execution occurred.

## Baseline comparison

| Baseline | Exact version space | Status accuracy | Sufficient exact | Ambiguity recall | Contradiction recall | Mean candidates |
|---|---:|---:|---:|---:|---:|---:|
| Complete safe enumeration | 0.0000 | 0.3333 | 0.0000 | 1.0000 | 0.0000 | 256.00 |
| Definition parser only | 0.2222 | 0.5556 | 0.6667 | 1.0000 | 0.0000 | 199.33 |
| Ontology retrieval only | 0.2222 | 0.5556 | 0.6667 | 1.0000 | 0.0000 | 199.33 |
| Observation filter only | 0.7778 | 0.7778 | 0.3333 | 1.0000 | 1.0000 | 25.00 |
| Parser plus exact version space | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 21.67 |
| Hidden-contract oracle | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 21.67 |

All baselines retained the target whenever the frozen evidence was noncontradictory, but the ablations show why
retention alone is insufficient. Enumeration never becomes informative; definitions alone cannot detect
contradiction or identify a table-defined provisional rule; observations alone underdetermine registered
definitions in two sufficient cells. The two evidence sources are complementary.

## Interpretation

Within this finite typed DSL, the hard open-world judgment can be decomposed into two explicit questions:

1. which candidate truth tables are consistent with the definition and interventions; and
2. whether the resulting version space has zero, one, or multiple members.

That factorization avoids the harmful instruction to choose the nearest catalog label. A provisional primitive is
identified only when the evidence uniquely determines a truth table outside the registered DSL, while ambiguity
remains a set and contradiction remains empty. This does not establish unrestricted natural-language
understanding or ontology truth; it establishes the model-free control for this bounded representation.

The zero residual means an LLM is not scientifically justified on V166. The next Track C experiment should make
the unresolved 64-candidate states sequential: preregister sensing actions, costs, exact belief/version-space
updates, and delayed decisions, then test whether a planner chooses useful evidence. Separately, Track B can
preregister a reversible fixed-ontology sandbox. Neither follow-on may run without its own lock.

## Frozen decision

Freeze V166 as positive project-authored model-free development evidence and advance to separately preregistered
evidence-gathering and reversible-sandbox designs, without an LLM residual.
