# V116 Typed Clarification Value-of-Information Results

## Outcome

V116 is a positive but strictly conditional design-feasibility result:

> `conditionally_feasible_requires_95pct_independent_typed_answers`

The audit used the frozen V115 structural population and candidate identifiers, the exact 17-choice
hypothesis universe, and frozen decision costs. It read no fresh or protected language, loaded or generated
with no model, emitted no individual diagnostics, retained every hypothesis, and executed nothing.

Two conditionally independent typed answers at 95% correctness cleared every preregistered feasibility gate
under uniform, moderately candidate-biased, and strongly candidate-biased priors. Fully correlated answers
failed the robustness test. This is simulator-conditioned value of information, not evidence that a real
human or model can supply answers with the required accuracy or independence.

## Primary 95% reliability result

The historical frozen V112-style policy on the V115 population had mean regret 0.7760, 76.04% exact known-
intent accuracy, 91.67% unsupported recall, and 5.21% false-known acceptance.

With two independent typed answers and a total clarification cost of 0.30:

| Prior | Mean regret | Known exact probability | Unsupported correct | False-known probability |
| --- | ---: | ---: | ---: | ---: |
| uniform safe universe | 0.7368 | 90.25% | 95.00% | 0.003% |
| moderate candidate | 0.7058 | 99.45% | 95.00% | 0.336% |
| strong candidate | 0.7174 | 99.45% | 90.25% | 0.336% |

All three priors beat the historical mean-regret baseline while clearing the frozen 80% known, 80%
unsupported, and 10% false-known requirements.

The boundary is sharp. At 90% correctness, the uniform-prior condition had 0.7972 regret and failed the
historical baseline. A single 95%-correct answer also had 0.7877 regret under the uniform and moderate
priors, narrowly failing. The second genuinely independent answer is what made the result robust across
the preregistered priors.

## Correlation stress test

When the two answers were fully correlated, repeating the answer supplied no additional evidence but still
incurred the larger 0.30 clarification cost. At 95% reliability:

| Prior | Fully correlated mean regret | Passes 0.7760 baseline? |
| --- | ---: | --- |
| uniform safe universe | 0.8877 | no |
| moderate candidate | 0.8877 | no |
| strong candidate | 0.7401 | yes |

The correlated condition therefore failed the all-prior robustness requirement. Asking the same source to
restate the same semantic judgment—as V115 effectively did—is not a valid substitute for a second
independent observation.

At a perfect channel, the two-answer policy reached 0.675 mean regret under every prior. It still abstains
on novel-valid requests, so its value comes from resolving known and unsupported actions safely rather than
granting authority to emit or register a novel capability.

## Interpretation and next gate

V116 establishes only that a high-reliability, independent typed answer channel could be decision-relevant
under the explicit simulator. It does not establish that two such answers exist in the current application.
In particular, two calls to the same LLM, two paraphrases, or two samples from closely related models must
not be treated as independent by assumption.

The result authorizes at most a separately preregistered, unprotected simulator benchmark that makes the
independence structure causal and explicit—for example, two orthogonal questions answered from separately
generated latent observations. That benchmark must stress correlated errors and misspecification and must
remain aggregate-only, nonauthoritative, and non-executable. It cannot use the original protected set or
claim human reliability.

Fresh language or model generation, protected access, schema induction, capability registration, richer
planning, APIs, training, action authority, and execution remain closed.
