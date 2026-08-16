# V55 results: exact short-horizon Bayes-adaptive planning

Decision: `do not advance; localize the failed sealed V55 gates`.

V55 evaluates exact three-action belief-space planning over program identity, continuous theta quadrature, and hidden world/queue configuration. It does not claim long-horizon, approximate, learned, language-grounded, or formally verified planning.

## Sealed results

- All qualification gates passed: `False` (19/20).
- Maximum primary/scalar root-value error: `0.0`.
- Maximum independent policy-evaluation error: `1.1102230246251565e-16`.
- Mean Bayes-adaptive value: `0.75409044921875`.
- Mean open-loop value: `0.73569921875`.
- Mean adaptive minus open-loop value: `0.01839123046874999`.
- Positive value-of-adaptation fraction: `0.21875`.
- Non-myopic root-action fraction: `0.15625`.
- Information-then-control fraction: `0.15625`.
- Delayed-consequence sensitivity fraction: `0.0`.
- Controls detected or dominated: `6/7`.
- Clairvoyant upper-bound violation rate: `0.0`.
- Integrity violations: `0`.

## Boundary

A full pass authorizes only a new preregistration for symbolic and probabilistic verification of the frozen finite-horizon policy. Formal verification itself, longer horizons, approximate search, language grounding, model access, and training remain blocked.
