# V214 deterministic candidate and version-space controls results

## Bottom line

V214 passed every frozen integrity, control, firewall, and access gate, and selected branch
`DETERMINISTIC_CLOSURE_ZERO_MODEL_ELIGIBILITY`.

The result is decisive for the current typed program representation: retrieval is unreliable, bounded representation
languages leave real gaps, but complete deterministic constraint propagation recovers the exact version space and
shadow decision for every development-evaluation record. A local LLM candidate generator is therefore not justified on
this branch.

## Prospective firewall

V214 reconstructed only the 80 V213 development groups from the frozen generator inputs. It did not open the V213
frozen public-record, sealed-truth, or split artifacts and did not construct protected group records.

The development groups were split by a frozen within-family hash ranking into 40 fit groups/160 fit records and 40
evaluation groups/160 evaluation records. Every family contributed four fit and four evaluation groups; all four
variants remained together and group overlap was zero.

The worker received fit public records, fit labels, and evaluation public records only. Predictions were hashed and
frozen before evaluation truth was joined. Its evaluation-truth path and hidden-evaluation-field counts were both zero.

## Control results

| Method | Oracle-class recall | Exact version space | False proposal rate | Action accuracy | Normalized regret | Residual groups |
|---|---:|---:|---:|---:|---:|---:|
| Exact structural ceiling | 1.0000 | 1.0000 | 0.0000 | 1.0000 | 0.0000 | 0 |
| Normalized exact retrieval K8 | 0.2000 | 0.3000 | 0.0000 | 0.4000 | 0.1775 | 28 |
| Typed approximate retrieval K8 | 0.2000 | 0.0000 | 0.9420 | 0.2000 | 0.0200 | 40 |
| Bounded `L0/L+` synthesis | 0.6375 | 0.6000 | 0.0000 | 0.7000 | 0.1025 | 16 |
| Full 256-class constraint propagation | 1.0000 | 1.0000 | 0.0000 | 1.0000 | 0.0000 | 0 |
| Deterministic stack | 1.0000 | 1.0000 | 0.0000 | 1.0000 | 0.0000 | 0 |

All methods emitted 160 predictions, respected their frozen candidate budgets, and were invariant across the four
presentations of every evaluation group.

## Interpretation

The controls separate three questions that top-1 accuracy would blur:

1. **Can a similar example supply the meaning?** Usually not. Exact retrieval missed most groups, while approximate
   retrieval proposed many incorrect behavioral classes.
2. **Can the frozen representation languages express the meaning?** Only sometimes. Bounded `L0/L+` synthesis was
   exact on 60% of records and safely made no false-class proposals, but it could not represent language-relative
   irreducibles and some underdetermined states.
3. **Does the public typed evidence determine a finite version space?** Yes. Enumerating and filtering all 256
   behavioral classes recovered every singleton, ambiguous pair, contradiction, and outside version space exactly.

The average candidate-set size of the exact methods was `26.5`, driven by outside descriptions retaining all 256
behaviors. This is intended version-space preservation, not uncontrolled proposal growth.

## Model eligibility

The frozen selector was the deterministic stack. It had:

- zero residual records;
- zero residual groups;
- zero average normalized decision regret;
- exact evidence status and action on all 160 evaluation records.

The preregistered local-model threshold required at least eight eligible residual groups and average normalized regret
of at least `0.02`. Neither condition was approached. The correct decision is therefore **not** to run a local or API
model on this typed representation.

This does not imply that LLMs are useless for open-world language. It shows that once meaning is already exposed as a
small executable evidence program, deterministic version-space inference is the right tool. Any future model study must
introduce a genuine semantic-projection problem—without discarding the exact deterministic back end—and must earn
eligibility on a new frozen residual.

## Access and claims

V213 protected public/truth access, protected construction, natural-language reads, external ontology payload reads,
model loading/generation, API calls, training, registration, trusted-state mutation, service calls, external side
effects, and actual execution all remained zero.

The result applies only to the project-authored typed program representation. It is not evidence about unrestricted
natural-language understanding, correct human meaning, or real-world ontology acquisition.
