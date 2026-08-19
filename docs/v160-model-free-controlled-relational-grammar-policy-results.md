# V160 model-free controlled relational-grammar policy results

## Result

V160 passed every prospectively locked development, selectivity, cost, firewall, and access gate.

The grammar-plus-retrieval router selected the exact initial action on all 64 development requests. It used
a specific question on all 16 lexical controls and all 16 unique registered-alias requests. It used generic
`Q80` on all 16 cross-query conflicts and all 16 insufficient requests. False-specific routing on the two
fallback strata was zero.

## Cost and controls

| Policy | Initial routing accuracy | Mean decision cost | Wrong specific questions | Final exact |
|---|---:|---:|---:|---:|
| Grammar + retrieval | 1.0000 | 0.2500 | 0 | 1.0000 |
| Information oracle | 1.0000 | 0.2500 | 0 | 1.0000 |
| Lexical margin | 0.6875 | 0.3250 | 8 | 1.0000 |
| Always generic | — | 0.3500 | 0 | 1.0000 |
| Source specific then generic | — | 1.0750 | 176 | 1.0000 |
| No query | — | 0.5000 | 0 | 0.5000 |

The grammar router improved cost by 0.25 over no-query abstention and by 0.10 over always-generic routing.
Its cost gap from the hidden information oracle was exactly zero.

## What the grammar fixed

The lexical comparator was exact on all lexical controls and insufficient requests. It missed 12 of 16
unique-alias requests by falling back generically, and it overcommitted on 8 of 16 cross-query conflicts by
selecting `Q83`. Those conflict errors were wrong specific questions, not harmless generic deferrals.

The finite grammar separated these cases mechanically:

- one or more known quoted aliases mapping to one query: ask that specific question;
- aliases mapping to different queries: ask generic `Q80`;
- any unknown quoted alias: ask generic `Q80`;
- no quoted surface: apply the unchanged strict retrieval rule.

Grammar therefore contributed a typed relation-cardinality check that a lexical score margin did not
provide. It did not infer a state or capability.

## Safety and access

All interactive comparators ended exactly because irrelevant specific questions produced no witness and
failed closed before generic fallback. Every irrelevant intermediate stayed `A00`; authoritative hypothesis
retention was 100%; candidate fields and execution were zero.

The run read 64 development requests and 128 locked development metadata rows. V159 evaluation access,
model loads or generations, API calls, training, real service calls, side effects, and execution were all
zero.

## Decision

Retain V160 as positive project-authored synthetic controlled-grammar development evidence. Do not open
V159 evaluation or tune the aliases, grammar, retrieval thresholds, costs, or gates. Do not introduce an
LLM merely to solve this now-deterministic interface.

The next authorized branch is only design of a fresh transfer population whose surfaces are not generated
from the V159 controlled templates. It should test whether independently supplied or externally styled
utterances can be translated into the finite relation codebook while preserving unknown-alias fallback.
That branch must be locked before any policy or model is run.

## Claim boundary

V160 does not establish general semantic parsing or unrestricted open-world language understanding. The
benchmark explicitly supplied quotation syntax and a registered alias codebook. The result proves that,
inside that controlled interface, deterministic grammar can recover the oracle question policy and is safer
and cheaper than lexical margin routing. External language remains an open problem.

