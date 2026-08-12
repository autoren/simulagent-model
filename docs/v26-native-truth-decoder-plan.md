# V26 protocol: full-depth native truth decoder

## Objective and status

V26 tests whether the frozen 4B model contains the required truth semantics at full depth even
though V25 showed that they are not linearly organized in layer-8 assessment-span features. It is
an exposed-data development experiment. Every V24 candidate assignment remains fixed, no head or
threshold is fitted, and no fresh benchmark record is constructed or read.

V25 is rejected rather than tuned: assessment-identity centroid distances were 1.83–2.27, while
within-assessment compatibility distances were only 0.10–0.13 for entailment and contradiction.
The shared head never selected unresolved on evaluation. V26 therefore changes the readout family,
not the labels, assignment, ontology, or executor.

## Native A/B/C readout

Each fixed evidence/candidate pair is rendered once. The system instruction defines exactly three
single-token outputs:

- `A`: evidence entails the candidate fact (`true`);
- `B`: evidence contradicts the candidate fact (`false`);
- `C`: evidence leaves the candidate fact unresolved (`unknown`).

The prompt runs through all 32 frozen transformer layers. The final prompt-token state is cast to
float32 and projected directly onto the dequantized embedding rows for `A`, `B`, and `C`. Argmax of
those three logits is the registered prediction. Native bfloat16 label logits are recorded only as
a numerical diagnostic and cannot select a method.

There is one forward pass per evidence unit (4,437 total), zero fitted heads, zero fitted
thresholds, and no calibration-based selection.

## Decision

Passing every development gate authorizes freezing the combined V24 matcher and V26 decoder before
constructing a fresh relational surface benchmark. Failure shows that the frozen model's native
decoder is not reliable enough for declared truth semantics; it does not authorize another linear
probe, LoRA, grammar expansion, or a final-suite claim. A later pivot must then compare a declared
semantic parser or a separately justified learned grounder on new development language.
