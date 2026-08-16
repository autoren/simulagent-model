# V43r1 preregistration: graph measurement repair

## Status of V43

V43 remains a sealed registered-gate failure. Its state-clause parsing, action parsing, safety behavior, mechanic recovery, next-state execution, final execution, and order counterfactuals were all exact, but direct list equality reported only 0.262 state-graph exactness.

A labeled post-hoc diagnostic found that all 1,147 graph pairs have identical canonical row sets, no duplicates, and no semantic content mismatches. The 847 mismatches arise because reference rows were sorted before canonical entity IDs were replaced by hashed aliases, whereas compiled rows were sorted after aliasing.

## Sole permitted repair

V43r1 replaces ordered-list equality with equality between duplicate-free canonical sets of `(atom, allowed_values)` rows. No V43 artifact, compiler output, stateful program, prediction, or other metric may change. The original V43 result and failed outcome lock remain immutable and continue to be reported.

The repair implementation must demonstrate permutation invariance. It must reproduce every non-graph V43 metric and every non-graph gate exactly from the sealed corpus. There is one repair rescore and no new corpus, selection, model access, or training.

## Decision

- Canonical graph exactness 1.000, duplicate-free 1.000, zero semantic mismatches, comparator permutation invariance 1.000, and exact reproduction of all other metrics: accept the V43 semantic graph result as measurement-repaired and preregister deterministic delayed effects.
- Any canonical content mismatch: reject the ordering-only diagnosis and revisit translation.
- Any other metric drift: reject V43r1 as a non-isolated repair.

V43r1 is a correction over the same paired development data, not an independent confirmation.
