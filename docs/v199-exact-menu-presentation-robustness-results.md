# V199 exact menu-presentation robustness results

## Verdict

V199 passes every preregistered feasibility and access gate. It freezes an exact paired challenge for testing whether
V195/V198 depends on incidental menu presentation.

This is a model-free design result, not language-robustness or model-performance evidence.

## Census

The frozen V191 development role contains 98 identities: 84 observed records, six per contract, and 14 missing
controls. Two deterministic variants were constructed for every identity, producing 196 record-variant menus and 168
observed record-variant cases:

- `ORDER_ONLY` changes record-keyed presentation order while preserving the canonical `M01`–`M14` map.
- `ORDER_AND_OPAQUE_ID` independently maps the contracts to `Q01`–`Q14` and changes presentation order.

Across all 196 menus:

- minimum and maximum option counts were both 14;
- semantic multiset preservation was `1.0`;
- hidden option-to-contract bijection rate was `1.0`;
- observed-target unique expressibility was `1.0`;
- changed presentation-order rate was `1.0`;
- canonical-ID preservation for `ORDER_ONLY` was `1.0`;
- exact opaque-ID-set rate was `1.0`; and
- visible forbidden-field count was zero.

The assignment function used only fixed salts, `record_id`, and canonical option IDs. It read no utterance,
conversation, truth kind, prior, score, model output, or error.

## Parser and controller

All 980 dynamic malformed, wrong-length, unknown-ID, duplicate-ID, and extra-key controls mapped to `INSUFFICIENT`.
The oracle top-3 question cost remains `0.20`, target retention remains `1.0`, and the trusted `OTHER` and complete
authoritative fallback remain unchanged.

## Frozen later gates

Before transformed scoring, V199 froze per-variant task and invariance gates: no more than 0.05 primary or macro
top-3 recall loss, no more than 0.02 primary cost increase, at least 0.80 top-1 contract agreement, at least 0.80 mean
top-3 contract-set Jaccard, no more than 5% target-inclusion disagreement, and at least 0.01 primary cost improvement
over `CHAR_LAST` on the same transformed menus. Structural validity must remain at least 0.98; final truncation and
false terminal decisions must remain zero; target retention and trusted exactness must remain one.

## Access and decision

Language reads and scores, model loads and generations, protected-language access, API calls, training, ontology
registration, trusted mutation, services, side effects, action, and execution were all zero.

Freeze:

`freeze_V199_exact_menu_presentation_robustness_family_and_authorize_separate_deterministic_development_evaluation_only`

The next permitted step is a separately preregistered deterministic development evaluation. V199 does not authorize
an immediate local-model run or protected access.

