# V9 symbolic evaluator preregistration

V9 first replaces learned transition algebra with an exact allowed-values
evaluator. Given the model-facing determinant order, complete transition table,
and a non-empty allowed value set for every determinant, it enumerates the
Cartesian product of compatible assignments, looks up their transition codes,
and declares the action identifiable exactly when one distinct code remains.

Before generating V9 language data, the evaluator must reproduce all 6,480
simulator-derived V8 development records with:

- zero identifiability mismatches;
- zero possible-transition-count mismatches;
- zero compatible-assignment-count mismatches; and
- complete agreement in every mechanic and surface cell.

Malformed schemas, missing determinants, duplicate determinants, empty value
sets, and missing transition cases are hard errors. This stage reads only the
already exposed V8 development corpus. It may not read Tone Drift, V3 test
records, prior holdouts, V7 model results, or any new/final mechanic.

Passing this audit authorizes V9 natural-language grounding data generation. It
does not authorize model extraction, LoRA, or final-mechanic evaluation.
