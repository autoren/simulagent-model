# V50 history-dependent belief filtering

## Why V50 is needed

V49 established the core passive-partial-observation capability. The exact system recovered every target program and probability, matched the target conditional future distributions to numerical precision, calibrated correctly, and failed catastrophically when its latent belief was collapsed to one state. It therefore demonstrated distributional hidden-state reasoning.

V49 did not demonstrate that earlier observations must be retained. Its generated one-event queries made the latest observation sufficient, so discarding earlier observations had no effect. V49 also gated the raw log-loss difference between partial and complete observation. That comparison was conceptually wrong: predictors with different information sets face different irreducible conditional entropy. V49's nearly zero TV errors in both conditions, alongside a 0.446-nat raw log-loss gap, show that the gap measured information value rather than model error.

V50 repairs both issues prospectively. It constructs histories for which the latest observation is not sufficient, and it compares conditions using regret relative to an oracle with the same information.

## Scope

The ontology, finite probability vocabulary, exact stochastic DSL, and sum-product belief kernel remain fixed. Language, active experiment selection, noisy or learned sensors, continuous probabilities, open ontologies, and neural training remain excluded.

The observation schedule is known and noiseless. Unlike V49's MCAR masks, V50 uses a structurally designed, value-independent schedule: an earlier step reveals branch- or queue-relevant evidence, a later prefix step hides it, and the suffix retains a consequence. The agent receives the actions and masks but does not choose them.

## Fresh population and construction firewall

V50 contains 48 fresh programs, balanced across the four stochastic families and three declared probabilities, with zero program overlap with V46–V49 and zero structural-case overlap with V47–V49. Each mechanic receives 12 support interventions with 32 trials and 24 query episodes with 64 held-out continuations.

Before the corpus can be sealed, at least 80% of query episodes must satisfy an oracle structural criterion: conditioning the target program on the complete observation history versus only the latest observation changes the correct suffix distribution by TV at least 0.10. This is a construction requirement, not a development result. It prevents another nominal history ablation in which the removed evidence is redundant.

## Inference and controls

The primary system is the frozen V49 exact joint filter over program, probability, world, and delayed-event queue, conditioned on the full masked history. The oracle-program reference validates state and queue filtering. The latest-only ablation removes earlier evidence, the time-shuffled ablation assigns valid observations to the wrong steps, and the MAP-state control collapses each filtered latent distribution to one configuration.

## Corrected scoring

Held-out conditional log loss remains the primary proper score within an information condition. Cross-condition model error is measured with condition-matched regret:

`model negative log probability - target-program oracle negative log probability`

Both terms receive exactly the same evidence. Partial-observation regret is compared with fully observed regret. Raw partial-minus-full log loss is reported only as the empirical value of additional information; it is not a performance gate. As a measurement check, that raw gap should agree with the corresponding oracle conditional-entropy gap within 0.03 nats.

## Gates

All distributions must normalize, log loss must remain finite, and every realized continuation must remain in support. Oracle-program mean TV must be at most `1e-10`; primary mean TV at most `0.05`; every-family mean TV at most `0.08`; and calibration error at most `0.05`.

Mean condition-matched regret must be at most `0.02` nats, and partial-minus-full condition-matched regret at most `0.01` nats. At least 80% of queries must be oracle-history-dependent, with mean oracle full-history versus latest-only TV at least `0.10`. Removing earlier observations must worsen log loss by at least `0.10` nats, shuffling observation time by at least `0.05`, and collapsing the latent belief by at least `0.01`.

Program recovery gates remain conservative: MAP recovery at least `0.80`, mean target posterior at least `0.70`, and probability MAE at most `0.07`. The mechanic episode remains the statistical unit and intervals use 10,000 mechanic-cluster bootstrap resamples.

## Decision

A full pass authorizes preregistration only of supported declared-language composition with history-dependent partial observation. If the construction audit cannot establish genuine oracle history dependence, the population must be repaired before any development run. If exact predictions pass but history controls do not, the project still may claim hidden-state marginalization but not temporal evidence retention.
