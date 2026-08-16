# V52 preregistration: Rao–Blackwellized particle filtering

## Why V52 is next

V51r1 independently calibrated the exact joint inference layer on 2,048 prior-predictive replications. All gates passed: the two exact implementations agreed to maximum TV `6e-100`, rank and coverage diagnostics passed, and the audit detected both the squared-likelihood and MAP-collapse controls. The exact filter is therefore suitable as an oracle, but its trajectory/configuration support grows exponentially with horizon.

V52 takes the next step in the report’s sequence: exact enumeration to Rao–Blackwellized particles. It retains exact summation over the finite 48 mechanic/probability hypotheses and approximates only the hidden dynamic world and delayed-effect queue. Each particle’s one-step stochastic branches are summed exactly before bounded resampling. This is the appropriate boundary for the current finite probability vocabulary; sampling static parameters or adding PMCMC/SMC² here would add approximation without necessity.

The design follows the principle of Rao–Blackwellized particle filtering—sample only the variables that need sampling and marginalize tractable variables—and uses the SMC² decomposition as the next-stage target: outer static uncertainty with an inner state filter. V52 deliberately stops before continuous parameters, static-parameter particles, or PMCMC rejuvenation.

## Three non-overlapping sealed populations

The exact benchmark has 96 fresh records over 48 fresh programs. Particle budgets 32, 128, and 512 are each run with three independent random streams and compared against the independently implemented exact stepwise filter. The targets are the support and query program posteriors, probability marginal, full joint program/world/queue belief, suffix predictive, and log evidence.

The SBC population has 1,024 longer prior-predictive replications and uses the precommitted primary budget of 512. It repeats V51’s five rank quantities with 63 posterior draws and frozen randomized tie handling. It is separate from the exact benchmark so particle choices cannot be selected on calibration outcomes.

The scale-stress population has 96 records with horizons 16, 24, and 32. Exact enumeration is not required there. Completion, normalization, target extinction, ESS, resampling, diversity, ancestry, bounded live support, proposal count, and runtime are recorded; runtime is descriptive rather than gating.

## Collapse, resampling, and random-stream integrity

Approximation must not silently convert uncertainty into certainty. On cases whose exact maximum program probability is at most 0.60, V52 measures false collapse to a particle posterior maximum of at least 0.95 and the ratio of approximate to exact program entropy. It separately detects loss of exact configurations carrying at least 0.05 posterior mass.

Every resampling stream is keyed by population, record, program, budget, repeat, episode, and tick. Unintended stream collisions are forbidden. Fingerprints must differ across independent repeats on stochastic resampling cases, and an intentional collision control must be detected. MAP-program, MAP-configuration, and squared-likelihood controls provide additional collapse and calibration sensitivity checks.

## Decision

All accuracy, convergence, SBC, degeneracy, stream-integrity, control-sensitivity, and scale gates are non-compensatory. A full pass authorizes only a preregistration for continuous stochastic parameters using SMC², with PMCMC as an offline reference. Active intervention selection, reward, planning, language, model calls, and training remain blocked.

References: [Doucet et al., Rao-Blackwellised Particle Filtering (UAI 2000)](https://research.google/pubs/rao-blackwellised-particle-filtering-for-dynamic-bayesian-networks/), [Chopin, Jacob & Papaspiliopoulos, SMC²](https://arxiv.org/abs/1101.1528), [Andrieu, Doucet & Holenstein, PMCMC](https://www.stats.ox.ac.uk/~doucet/andrieu_doucet_holenstein_PMCMC.pdf), and [Talts et al., Simulation-Based Calibration](https://arxiv.org/abs/1804.06788).
