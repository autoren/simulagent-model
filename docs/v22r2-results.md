# V22r2 results: hard relational-language grounding

Decision: `develop_probabilistic_support_interface_no_lora`. Development gates passed: `false`.

V22r2 is an open development result. The V22 symbolic oracle remains exact, but the fixed
hard language interface does not yet populate it reliably from held-out wording.

## Registered result

| Condition | Transition-set exact | Target retained | Empty version spaces |
|---|---:|---:|---:|
| `frozen_support_frozen_query` | 0.167 | 0.250 | 0.333 |
| `frozen_support_oracle_query` | 0.192 | 0.250 | 0.333 |
| `oracle_support_frozen_query` | 0.590 | 1.000 | 0.000 |
| `oracle_support_oracle_query` | 1.000 | 1.000 | 0.000 |

## Held-out grounding

- atom assignment: 0.814;
- ordered-relation assignment: 0.639;
- truth status: 0.782;
- exact scene graph: 0.031; and
- all-support episodes exact: 0.000.

## No-refit diagnostic

Evaluation support gold-atom retention rises from 0.730 at top 1 to 0.937 at top 2 and 0.972 at top 3.
Evaluation query gold-atom retention rises from 0.737 at top 1 to 0.907 at top 2 and 0.960 at top 3.

Component-oracle downstream exact match:

| Component condition | Support / oracle query | Oracle support / query |
|---|---:|---:|
| Fully frozen | 0.192 | 0.590 |
| Oracle atom assignment, frozen truth | 0.237 | 0.712 |
| Frozen atom assignment, oracle truth | 0.699 | 0.795 |

## Interpretation

Support certainty is the immediate integration bottleneck, but both atom alignment and held-out
truth phrasing contribute. A probabilistic support experiment is justified only as a controlled
decomposition: it should preregister small top-k branch budgets and require target retention gains
without accepting broad answer sets as success. Query grounding must remain a separate reported ceiling.
No LoRA, final suite, grammar expansion, or joint neural challenger is authorized by this result.

The first evaluation attempt aborted before any prediction because the installed scikit-learn
required an explicit one-vs-rest wrapper for multiclass liblinear. V22r2a locked that
nondiscretionary compatibility amendment before the single completed replacement run.
