# V49 passive partial observation

## Purpose and claim

V48 established lossless composition between the supported declared-language compiler and the sampled stochastic symbolic estimator. V49 makes one architecture-breaking change: post-action states are no longer completely visible. It tests whether the structured system can preserve a posterior over the hidden world and delayed-event queue while simultaneously inferring the stochastic program and its probability.

The claim is deliberately restricted to a declared Boolean ontology, a finite frozen program registry, the known probability vocabulary `1/4`, `1/2`, and `3/4`, and public noiseless masks sampled independently of the latent values. V49 does not test language, unknown observation models, noisy sensors, active experiment choice, continuous probabilities, open ontologies, or neural training.

## Why language is removed again

Partial observability introduces a qualitatively new uncertainty source. Keeping the input symbolic localizes failures to belief-state inference instead of confounding them with parsing. V48 already established that the supported language interface can preserve fully observed stochastic inference exactly. A V49 pass can authorize a separate paired language-composition experiment afterward.

## Fresh population

The population contains 48 fresh mechanics, 12 from each of the four stochastic families and 16 at each declared probability. No V46, V47, or V48 program or structural case may recur. Twenty-four mechanics are labeled development-fit and 24 development-evaluation, but neither subset may be used for per-mechanic choices.

Each mechanic receives 12 fixed support interventions and 32 independent trials per intervention. A trial starts from a completely known world. After each action, the agent sees only the Boolean values selected by a public step-specific mask. Masks reveal 25%, 50%, or 75% of atoms, are independent of the program and realized values, and contain at least one visible and one hidden atom.

There are 24 query episodes per mechanic. Each query supplies a complete initial world, its actions, and a sampled masked prefix of one to three steps. The target is the conditional distribution over the complete latent suffix trajectory. Query prefixes must retain genuine hidden-state ambiguity, and the hidden state must be capable of affecting the scored suffix in at least 75% of queries. Conditional query outcomes and exact oracle distributions are scorer-only.

## Exact belief inference

For each candidate program, support likelihood sums the exact probability of every full trajectory compatible with the observed masks. Query inference then conditions the support posterior on the query's masked prefix. The latent configuration includes both the complete current world and the delayed-event queue; omitting the queue would make delayed mechanics incorrectly Markovian in the visible world alone.

The primary prediction marginalizes over program, probability, latent world, and queue. The output is therefore a distribution rather than a guessed completion. Aleatoric trajectory entropy and epistemic program uncertainty are reported separately as diagnostics.

## Comparisons and controls

The matched fully observed condition uses the same mechanics, trials, actions, queries, and random draws but exposes all post-action state values. It quantifies the cost of missing observations. The oracle-program condition fixes the correct program while retaining hidden-state filtering; its near-exact TV gate validates the belief filter itself.

The MAP-state control collapses the filtered configuration posterior to one world-and-queue state. The history ablation discards all but the latest visible query step. Uniformized outcome mass removes learned probability structure, while literal trace lookup tests non-lifted memorization. These controls must be inadequate for a positive belief-state claim.

## Gates

All likelihoods, filtered beliefs, and conditional forecasts must normalize; all held-out log losses must be finite; and the realized latent continuation must always remain in the predicted support. Oracle-program partial-observation TV must be at most `1e-10`.

For the primary joint inference condition, mean conditional latent-suffix TV must be at most `0.08`, every-family mean at most `0.12`, and calibration error at most `0.07`. MAP schema recovery must be at least `0.80`, mean target-program posterior at least `0.70`, and probability MAE at most `0.07`. Relative to matched full observation, mean TV and held-out log loss may each degrade by at most `0.05`.

Collapsing the latent belief must worsen log loss by at least `0.01` nats, discarding observation history by at least `0.005` nats, and uniformizing predicted outcome mass by at least `0.03` nats. Literal support-query trace coverage may be at most `0.05`. Intervals use 10,000 mechanic-cluster bootstrap resamples; the mechanic episode is the statistical unit.

## Decision

A full pass authorizes preregistration only of supported declared-language composition with passive partial observation. Failure of the oracle-program reference is an implementation or transition-semantics failure. If the oracle reference passes but joint inference fails, the next step is to revise identifiability or passive evidence coverage, not to add language or LoRA. Active intervention selection, learned/noisy sensors, open ontologies, and continuous probabilities remain separate later stages.
