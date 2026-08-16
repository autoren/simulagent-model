# V53 preregistration: continuous-parameter SMC² with PMCMC reference

## Purpose and boundary

V52r2 established that exact finite program/probability enumeration can be Rao–Blackwellized with bounded particles over the hidden world and delayed queue while retaining exact-oracle agreement, SBC calibration, ambiguity, and stream integrity. V53 now introduces one genuinely continuous shared stochastic parameter, \(\theta\), and nothing downstream.

The target posterior is

\[
p(s_t, M, \theta \mid h_t),
\]

with eight finite program templates \(M\) enumerated exactly, continuous \(\theta\in[0.05,0.95]\) represented by outer particles, and hidden world/queue state represented by an inner particle filter. This follows the report’s recommended `exact → Rao–Blackwellized particles → SMC²/PMCMC` sequence. Active intervention selection, reward, planning, verification, language, model calls, training, noisy sensors, and open ontologies remain excluded.

## Three independent inference views

1. **Exact small-case oracle.** A 257-node Gauss–Legendre rule integrates the scaled Beta(2,2) prior over \(\theta\); at each node, the world and delayed queue are summed by exact stepwise enumeration. This yields program weights, the continuous parameter marginal, binned joint \((M,\theta)\), configuration belief, suffix predictive, and evidence.
2. **Online SMC² system.** Each exactly enumerated program maintains outer \(\theta\) particles. Each parameter particle owns an inner hidden-state filter. Outer systematic resampling occurs below 0.5 ESS and is followed by two fixed-scale particle-marginal Metropolis–Hastings moves. No proposal adaptation or result-dependent particle count is allowed.
3. **Offline PMCMC reference.** Four independent PMMH chains check \(p(\theta\mid M,h)\) for the generating program on 16 preselected exact-benchmark records, using 509 inner particles. This reference does not estimate program evidence; full joint validation remains the job of the exact quadrature oracle.

SMC² is the natural online method here because it nests a latent-state particle filter inside particles over static parameters. PMCMC is retained as an offline algorithmic cross-check, as recommended in the report and in the original SMC² and PMCMC work: [Chopin, Jacob & Papaspiliopoulos](https://arxiv.org/abs/1101.1528) and [Andrieu, Doucet & Holenstein](https://www.stats.ox.ac.uk/~doucet/andrieu_doucet_holenstein_PMCMC.pdf).

## Populations and firewalls

The exact population has 32 records over eight fresh parameterized templates. The prior-predictive SBC population has 256 replications. Sixteen even exact-record ordinals are fixed for PMCMC before population construction. The scale population has 32 records with horizons 24, 40, and 64. Every observation design is globally disjoint and fresh against V46–V52. Implementation fixtures add one million to every seed and cannot access any sealed candidate.

The shared parameter is drawn once per record and governs the same stochastic branch across all support and query episodes. That is essential: V53 tests static-parameter learning, not unrelated per-step probabilities.

## Non-compensatory validation

The exact benchmark measures program TV, Wasserstein distance for \(\theta\), binned joint \((M,\theta)\) TV, configuration TV, suffix-predictive TV, and log-evidence error at outer budgets 31, 127, and 509. Error must improve with budget. SBC uses ranks for program identity, continuous \(\theta\), current configuration, and posterior probabilities. PMCMC must independently satisfy acceptance, split-Rhat, bulk-ESS, and Wasserstein gates.

Degeneracy gates reject extinction of the true program, false collapse of ambiguous programs, false collapse of broad continuous posteriors, low outer ESS, and loss of parameter ancestry. Every outer and inner random stream is keyed by population, record, program, budgets, repeat, parameter particle, episode, tick, purpose, and move. Stream and resampling-fingerprint collisions are forbidden except in an intentional collision control.

MAP-program, theta-point-mass, squared-likelihood, outer-resampling-disabled, and deliberate-stream-collision controls test sensitivity. All gates are non-compensatory.

## Decision

A full pass authorizes only a preregistration for exact one-step expected information gain over the finite intervention set. It does not authorize active-population construction or any reward-bearing planner. Any failure stays within inference: repair SMC², PMCMC, degeneracy, calibration, or stream partitioning according to the frozen hierarchy.
