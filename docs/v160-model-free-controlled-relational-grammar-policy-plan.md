# V160 model-free controlled relational-grammar policy plan

## Question

Can a finite, public relation-alias grammar distinguish unique query evidence from cross-query conflict
without treating a lexical score margin as semantic confidence?

## Locked policy

The policy receives only visible development request text and a state-free projection of the four public
specific questions. Each projected question contains its title, question text, option text, retrieval
profile, and two registered relation aliases. It contains no choice ID, state ID, witness, compatible set,
truth label, oracle query, effect, precondition, or executable field.

The policy applies this fixed order:

1. Unicode-normalize the visible request and extract all double-quoted surfaces.
2. If any quoted surface exists, normalize each surface and look it up in the registered alias map.
3. Ask a specific question only when every quoted surface is registered and all aliases map to exactly one
   query.
4. Use generic `Q80` if an alias is unknown, no registered alias remains, or aliases map to multiple
   questions.
5. Only when no quoted surface exists, apply the unchanged strict retrieval rule: weights 8/3/1/0.25,
   minimum top score 6, minimum margin 4, and a unique top score.

No parameter is fitted. Grammar takes precedence over lexical retrieval so a lexically strong alternative
cannot overwrite an explicit cross-query conflict.

## Comparators

- safe no-query abstention;
- source-order specific questions followed by generic fallback;
- lexical margin routing without grammar;
- always-generic routing;
- hidden information oracle; and
- grammar-plus-retrieval routing.

Wrong specific questions are non-authoritative. Their answer yields no witness and the controller falls
back through `Q80`. Generic routing also yields no witness. Only a valid answer to the routed specific
question may update final state.

## Noncompensatory gates

The grammar router must achieve:

- exact initial action on all 64 development requests;
- 100% specific routing on lexical controls and unique registered aliases;
- 100% generic routing on cross-query conflict and insufficient requests;
- zero false-specific routing on grammar fallback strata;
- mean decision cost at most 0.25;
- improvement at least 0.25 over no-query and 0.10 over always-generic;
- zero cost gap from the information oracle;
- exact final outcomes, complete retention, and fail-closed irrelevant intermediates;
- zero candidate fields, evaluation access, model/API calls, training, side effects, or execution.

An explicit unit contract additionally requires an unknown quoted alias to route generically.

## Claim boundary

Passing would establish only finite controlled-grammar feasibility on fresh project-authored synthetic
development language. It would not establish general semantic parsing, natural-language robustness,
external transfer, unrestricted open-world recognition, or deployment safety.

