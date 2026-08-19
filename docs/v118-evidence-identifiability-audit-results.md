# V118 Evidence Identifiability Audit Results

## Outcome

V118 passed as an aggregate algebraic audit:

> `identifiability_boundary_derived_requires_asymmetric_adaptive_evidence`

It used only the frozen V117 hypotheses, costs, priors, channel, structural labels, and candidate identifiers.
It introduced no new simulator channel, language, model, prompt, protected access, training, or execution.
All 17 hypotheses remained available and no record-level result was emitted.

The audit explains both major V117 failures. The two questions did not supply the same kind of missing
information: exact known action requires much stronger evidence about candidate identity, while safe
unsupported action needs a small amount of additional evidence specifically against the strong known
candidate. A general repeated clarification protocol is therefore the wrong architecture.

## Frozen posterior requirements

Under the frozen decision costs, an exact candidate action becomes preferable to abstention only when its
posterior is approximately 89.47%--90.91%, depending on what kind of hypotheses hold the remaining mass.
An unsupported action requires approximately 71.43%--83.33% posterior mass.

Those posterior boundaries imply these Bayes-factor ranges:

| Prior regime | Exact candidate BF required | Unsupported BF required |
| --- | ---: | ---: |
| uniform safe universe | 136--160 | 40--80 |
| moderate candidate | 8.5--10 | 77.5--155 |
| strong candidate | 2.83--3.33 | 157.5--315 |

The requirements reverse with the prior. A diffuse prior makes exact candidate identity hard but makes
unsupported status easier. A strong candidate prior makes candidate acceptance easy but requires much
stronger counterevidence before declaring the request unsupported.

## Why V117 could not clear the uniform-prior gate

For the decisive `CONFIRM + DECLARED` observation at 95% marginal reliability, the frozen V117 Bayes factor
for the candidate against another known intent was:

| Shared failure correlation | Bayes factor | Information |
| ---: | ---: | ---: |
| 0.00 | 38.00 | 5.25 bits |
| 0.25 | 51.33 | 5.68 bits |
| 0.50 | 78.00 | 6.29 bits |

All are below the minimum uniform-prior requirement of 136. Catalog status says `DECLARED` for both the
candidate and every other known intent, so it contributes no candidate-versus-other-known separation.
This exactly explains the 0% known accuracy under the uniform prior through correlation 0.50.

One independent 95%-reliable exact-confirmation unit supplies a Bayes factor of 38 in the frozen error
geometry. Two genuinely independent identity-specific units would supply 1,444, exceeding the conservative
160 requirement. This is a mathematical requirement, not evidence that two real independent mechanisms
currently exist.

## Why the strong prior suppressed unsupported action

For the decisive `REJECT + OUTSIDE_VISIBLE` observation under the strong candidate prior, the
correlation-aware posterior changed as follows:

| Correlation | P(unsupported) | P(known) | Best action | Extra unsupported-specific BF needed |
| ---: | ---: | ---: | --- | ---: |
| 0.00 | 83.24% | 13.43% | unsupported | 1.00 |
| 0.25 | 81.08% | 16.34% | abstain | 1.091 |
| 0.50 | 79.08% | 19.02% | abstain | 1.270 |

Shared-failure modeling leaves enough probability on the costly known alternatives that abstention becomes
safer. The gap is small: an additional independent observation that favors unsupported over the rest by a
Bayes factor of at most 1.27 would cross the decisive boundary. Again, this describes required evidence; it
does not license fabricating or assuming such evidence.

## Structural ceiling and next gate

The frozen candidate is exactly correct for 93 of 96 known records, so any protocol that can only confirm
or reject that candidate has a 96.875% perfect-channel known ceiling. That ceiling is adequate for the
current 80% gate, but rejected known candidates require abstention because neither V117 question identifies
which other known intent is correct.

The next permissible branch is a separately locked, language-free adaptive causal simulator. It should ask
identity-specific evidence when the candidate needs confirmation and support-specific evidence when the
candidate is rejected or the catalog boundary is implicated. Its mechanisms must be causally distinct,
with explicit correlation and misspecification stress; a second wording, resample, or same-model judgment
cannot be labeled independent. No language/model run is authorized by V118.
