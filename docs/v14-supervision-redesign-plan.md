# V14 preregistration: operator-supported grounding corpus

## Motivation

V13 found a strong 4B token-local polarity representation but a single exact inversion: Direct Assertion scored 0.000 accuracy/AUC when held out. The post-result support audit showed that Direct Assertion was the only `gold_only` family, while Contrastive Correction was the only `both` family. Those folds removed an entire semantic operator signature, not merely one surface realization.

V14 corrects that taxonomy before any additional model training. It preserves the V10 semantic contexts, complement-disjoint train/evaluation split, intervention design, state lexicons, action schemas, symbolic evaluator, and protected-data firewall.

## Factorized current-evidence language

V14 has three semantic operator signatures, each with three independently worded surface families:

| Semantic operator | Literal mention orientation | Surface families |
| --- | --- | --- |
| `affirmative_gold` | gold hypothesis only | `direct_assertion`, `present_confirmation`, `current_observation` |
| `negated_opposite` | opposite hypothesis only under negation/rejection | `explicit_negation`, `denied_claim`, `scoped_rejection` |
| `contrastive_both` | both hypotheses, with opposite rejected and gold affirmed | `contrastive_correction`, `contrastive_verification`, `contrastive_resolution` |

Every current-evidence surface must exactly realize its registered mention orientation. Every surface-family holdout must retain at least two training surfaces with the same semantic operator. Current active/inactive targets remain exactly balanced inside split × mechanic × surface × lexicon cells.

Non-current evidence retains the locked UNKNOWN_CURRENT, STALE_ONLY, and CONFLICTING_CURRENT meanings and exact symbolic consequences. V14 records explicitly carry `semantic_operator_family` for auditing only; it is not included in model prompts.

## Corpus and pre-model gates

The 540 V10 source scene/surface/intervention records expand over nine current-evidence surfaces to 4,860 records. Expected structure:

- 90 complement-isolated semantic context groups, each with 54 records;
- 810 template-specific intervention groups, each with six records;
- three state lexicons, six mechanics, and two transition-operator families;
- exact spans, complementary hypotheses, derived allowed values, and symbolic outputs;
- no duplicate or cross-split prompts, literal target labels, or determinant IDs in observations.

Before any model access, V14 must pass structural validation, the existing metadata/position shortcut ceilings, and a new hard operator-support gate requiring every primary surface holdout to retain its evaluation mention signature in training.

## Evaluation split semantics

Primary generalization folds are context, mechanic, surface family, lexicon, transition operator, and operator×lexicon combined folds. Surface-family folds test paraphrase transfer with semantic-operator support.

Three semantic-operator holdouts are reported separately as zero-shot diagnostics. They are not part of the primary pass gate and cannot be described as surface transfer. No zero-shot result may be used to tune the primary head.

## Model sequence after the data gate

If the corpus and shortcut audit pass, the first model evaluation will reuse the pinned Qwen3.5-4B revision and layer 8/32. The primary polarity representation will be V13's hypothesis-token mean signed comparison. No 0.8B/9B comparison, alternate layer, MLP, or LoRA is permitted in that first V14 model run.

Only a separately frozen protocol may authorize extraction and evaluation after the corpus gate. V14 corpus construction itself permits no model access.

## Firewall

V14 corpus development reads only the already exposed V8 development source and the exact symbolic evaluator. Access remains zero for V3 test records, prior holdouts, V7 Tone Drift, V7 model results, the untouched V8 mechanic, and the final V9 mechanic. Adapter training and final-mechanic evaluation remain forbidden.
