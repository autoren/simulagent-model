# V51 simulation-based calibration

## Why V51 is next

V50r1 established exact history-dependent passive belief filtering: same-information regret was zero, every query required earlier evidence, latest-only prediction lost 0.516 nats on average, and the V49 cross-information scoring error was repaired. That is a positive task result, but it is not yet an independent calibration audit of the inference implementation.

V51 follows the prior-predictive workflow of Talts et al. and the Stan SBC guide: draw latent quantities from the prior, simulate observations from the declared likelihood, compute the posterior, draw independently from that posterior, and test whether the simulated latent ranks are uniform. Because the V51 latents are discrete, ties use a frozen randomized insertion rank. There are 63 posterior draws, hence 64 possible ranks, grouped into 16 equal bins.

## Scope

V51 remains exact and language-free. It introduces no particle approximation, active intervention selection, reward, planning, noisy sensor, continuous probability, model call, or training. It uses 2,048 fresh independent prior-predictive replications over a fresh balanced registry of 48 programs.

The primary inference path uses the existing exact batch trajectory enumeration. The independent reference performs sequential configuration filtering and stepwise observation updates without constructing a full trajectory catalog. Their program posterior, query-updated program posterior, joint program/world/queue belief, and suffix predictive must agree to TV at most `1e-12`.

## Calibration and sensitivity

SBC test quantities include canonical program ordinal, probability ordinal, canonical latent-configuration ordinal, and posterior probabilities assigned to the simulated program and configuration. Rank uniformity is assessed with a 16-bin chi-square test and maximum standardized bin deviation. Central posterior-set coverage is checked at 50%, 80%, and 95%.

A calibration pass is only meaningful if the audit detects known bad inference. Three frozen controls temper the likelihood, discard earlier query evidence, or collapse the posterior to its MAP value. At least two must be rejected on the same replications.

## Decision

A full pass authorizes preregistration of scalable Rao-Blackwellized particle inference against these exact references. Any exact-path disagreement or calibration failure blocks approximation. Active EIG remains downstream of the scalable-inference gate.

References: [Talts et al. (2018)](https://arxiv.org/abs/1804.06788), [Stan SBC workflow](https://mc-stan.org/docs/2_39/stan-users-guide/simulation-based-calibration.html).
