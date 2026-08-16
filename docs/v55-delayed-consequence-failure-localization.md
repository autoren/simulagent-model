# V55 delayed-consequence gate: failure localization

Decision: retain the sealed V55 failure and preregister a separate delayed-consequence adequacy confirmation.

V55 passed 19 of 20 gates. The failed gate is not explained by an off-by-one transition or Bellman error. The frozen delay fixture proves that a delay-two event scheduled at tick zero is delivered before the third action, while the primary planner, scalar reference, and independent policy evaluator agree.

The failure is localized to task support. Only one of eight latent templates contains a delay-two branch. Only 8 of 32 goals concern the affected `active` predicate, and only one task combines an `active` goal with the delay-two template as generating truth. In the sole delay-two template, the stochastic delayed `set_false(active(target))` effect also has a deterministic immediate same-target `route` substitute. The original fixture established temporal visibility but never established that delay could alter an optimal action or value. Consistently, removing every delay-two effect changed the root value by exactly zero on all 32 tasks.

The repair must not rerun V55. A new preregistration may define a separate confirmation suite with several delay-two latent templates, no same-target immediate substitutes, truth-independent goal stratification over delayed targets, and exhaustive decision-relevance fixtures. Formal verification remains blocked until that confirmation passes.
