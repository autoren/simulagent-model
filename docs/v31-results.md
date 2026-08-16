# V31 matched-head signed-fact adaptation

## Verdict

The frozen readout failed and the LoRA readout failed the preregistered language gates. Selected system: `none`.

## Sealed evaluation

| System | Predicate | Arg 1 | Relation order | Truth | Exact fact | Exact scene | Pass |
|---|---:|---:|---:|---:|---:|---:|:---:|
| Zero-shot reference | 0.985 | 0.928 | 0.376 | 0.567 | 0.341 | 0.090 | reference |
| Frozen readout (mean) | 0.967 | 0.980 | 0.961 | 0.814 | 0.783 | 0.617 | no |
| LoRA readout (mean) | 0.222 | 0.508 | 0.254 | 0.493 | 0.103 | 0.033 | no |

## Attribution

LoRA minus frozen exact-fact delta: -0.680.
LoRA minus frozen exact-scene delta: -0.583.
Family-bootstrap 95% interval for exact-fact delta: [-0.764, -0.587].
Material LoRA advantage: `false`.

## Decision integrity

One V28 replay authorized: `false`.
Seed selection: `none`.
Checkpoint or hyperparameter selection: `none`.
Post-result integrity audit: `pass`.
