# V209/V209r1 controlled probabilistic language-observation POMDP result

## Verdict

V209r1 is a positive model-free mechanism result. V209 itself stopped without a scientific result because its kernel validator incorrectly rejected the preregistered one- and two-regime comparator kernels. V209r1 changed only that shape invariant, reused the immutable V209 config payload, and passed every locked scientific and access gate.

The result supports this narrow claim:

> In a finite, explicitly specified probabilistic language channel, exact open-world planning can use clarification adaptively, preserve an outside-semantics hypothesis, act after informative language, and safely defer after unresolved language better than fixed or collapsed alternatives.

It is not evidence that an LLM understands the surface utterances or supplies calibrated likelihoods.

## Exact policy

The exact policy uniquely selected `ask_reference` at the root with value `2.0268896`. Its root alternatives were:

- `defer`: `-2.0`
- `ask_target`: `-2.5`
- `act_A`: `-10.0`
- `act_B`: `-10.0`

After a reference utterance:

- `UTTERANCE_ALPHA` selected `ask_target`;
- `UTTERANCE_BETA` selected `ask_target`; and
- `UTTERANCE_UNRESOLVED` selected `defer`.

Across later histories, both `act_A` and `act_B` were reachable. Three histories ended in safe deferral.

## Comparator results

The normalized value scale was `120`. Exact planning beat immediate deferral and the best open-loop program by `0.0335574133` normalized units. The best open-loop program was simply `defer`.

Normalized regret under the full true mixture was:

- closed-world Bayes-adaptive: `0.0149885467`;
- forced commitment: `0.0149885467`;
- MAP certainty equivalence: `0.1043907467`;
- persistent posterior sampling: `0.0831407467`; and
- myopic immediate-reward control: `0.10022408`.

The closed-world policy asked the target after an unresolved reference reply, while the full policy deferred. MAP and the two known-regime posterior-sampling policies asked the target at the root and had true-environment value `-10.5`; the outside-regime point policy deferred.

## Structural and safety checks

The reference response contained `0.5192531842` nats of mutual information about the semantic regime. The two known target channels had total variation `0.9`. History changed target likelihoods by up to `0.005`, and latent-dependent clarification costs spanned `0.012`.

All distributions remained normalized with minimum support probability `0.02`. Every visited belief normalized, and no fallback occurred.

All seven terminal paths were accountable:

- four automatic state-dependent settlements;
- three safe deferrals;
- zero unsettled controls; and
- zero horizon escapes.

The direct and matched-paraphrase surface families produced zero recovered-policy mismatches and exactly zero value difference. Swapping the `ALPHA` and `BETA` observation labels, permuting all corresponding probability axes, and mapping branch keys back also produced zero policy mismatches and zero value difference.

## Technical-repair record

Before V209r1, all six V209 unit tests passed. The first V209 oracle attempt computed the full exact policy in memory but failed while constructing the closed-world comparator, before any comparator value, regret, scientific gate, summary, or result was produced. The failure is preserved in `outputs/v209-controlled-language-observation-pomdp/technical-failure.json`.

V209r1 validated the regime dimension dynamically while continuing to require exactly two task states, three observations, normalized finite channels, common positive support, correct cost shapes, and fixed history anchors. One-, two-, and three-regime kernels then constructed successfully. No hypothesis, grammar surface, likelihood, cost, stage, reward, comparator, gate, or decision rule changed.

## Boundary and next gate

No external language record, model response, protected record, local model, API, training run, ontology registration, trusted-state mutation, service call, side effect, action, or execution was used.

The positive result authorizes only a separately preregistered fresh controlled-language population and deterministic semantic-observation projection. That successor must keep semantic truth separate from surface realization, use held-out surface constructions and matched counterfactuals, and freeze population generation and scoring before opening records. It still does not authorize a model run.
