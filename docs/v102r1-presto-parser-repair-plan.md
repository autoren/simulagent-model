# V102r1 PRESTO Parser Repair Plan

## Scope

V102r1 repairs only the technical mismatch frozen by V102. The source archive identity, en-US and human-
context restrictions, target-argument dependency rule, eligible context sources, development/test split,
scientific counts, diversity thresholds, and zero-model boundary remain byte-for-byte those in the
original V102 scientific config.

## Repair

Context containers must remain lists of objects where applicable, but null containers may be treated as
empty. Only actual string leaves may supply dependency evidence. A missing, null, numeric, Boolean,
object, or list leaf where the README describes a string is ignored and counted; it is never coerced to
text, tokenized, hashed, emitted, or manually inspected. At least one admissible target argument must
still be absent from the current input and occur contiguously in an actual string context leaf.

The verified archive is persisted before parsing. This ensures that another purely technical schema
error can be repaired against the same immutable local bytes without another network retrieval.

## Boundary

V102r1 may perform one fresh download because V102 persisted no response artifact. It may automatically
parse language but may not emit or manually inspect it. Passing still authorizes only a later text-free
paired-population preregistration, not language extraction, model access, API use, training, planning,
or execution.
