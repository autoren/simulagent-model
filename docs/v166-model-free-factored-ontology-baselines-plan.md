# V166 model-free factored ontology baseline plan

## Purpose

V165 established that expressibility relative to a finite DSL and evidence status can be represented separately.
V166 compares deterministic controls before any model is considered. The key output is an exact version space,
not a forced label.

## Baselines

Six frozen strategies run on all 144 development-only records:

1. complete safe enumeration retains all 256 candidate truth tables;
2. the controlled definition parser uses only exact registered definitions;
3. ontology retrieval exactly matches canonical registered definition strings;
4. observation filtering ignores definitions and keeps every truth table consistent with interventions;
5. parser plus exact version-space filtering combines both evidence sources; and
6. an oracle returns the hidden version space for scoring only.

Candidate order is deterministic. Cardinality zero means contradictory, one means sufficient, and more than one
means ambiguous. An ambiguous version space is never ranked or pruned, and contradiction is never automatically
repaired.

## Gates and interpretation

The combined exact method and oracle must reproduce every hidden version space, status, sufficient target and
expressibility class, ambiguity, contradiction, and renaming relation exactly. Target retention must be complete;
false provisional creation and false resolution must be zero.

The model-eligible residual is defined prospectively as any record where the exact combined public computation
differs from the hidden version-space contract. Intentionally ambiguous records are not model residuals: they are
correctly unresolved evidence states. If the residual is zero, no LLM is scientifically justified. The next step
is to design information-gathering actions that reduce the version space and a separate reversible fixed-ontology
sandbox—not to ask a model to guess among observationally equivalent candidates.

## Boundary

V166 is project-authored finite-DSL development evidence. Evaluation data, manual judgments, models, APIs,
training, provisional registration, services, side effects, authority, action, and execution remain zero. Passing
authorizes only separate prospective designs for the evidence planner and fixed-ontology sandbox.
