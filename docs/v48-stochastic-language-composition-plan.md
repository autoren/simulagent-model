# V48 stochastic language composition

## Purpose

V45 validated the declared language compiler for deterministic delayed mechanics. V47 validated finite-sample stochastic inference from symbolic realized trajectories. V48 composes those frozen components on a fresh population. It asks whether language can describe initial states, ordered actions, and realized post-action states without materially degrading stochastic mechanic induction or calibrated prediction.

This is not probability-language grounding: probabilities are never stated in text. They must be inferred from repeated realized outcomes.

## Population and matched comparison

The 48 fresh programs are disjoint from V46 and V47, balanced across four families and three probability values, and split evenly between development-fit and development-evaluation mechanics. Each mechanic has 12 support interventions with 32 realized trials and 24 disjoint queries with 64 scorer-only held-out trials.

The primary condition receives only supported V45 state and action language plus language descriptions of realized support trajectories. A matched symbolic condition receives exactly the same programs, worlds, actions, trial draws, and queries after correct compilation. This paired baseline localizes any difference to the language interface.

Fresh entity aliases, predicate cues, and action cues are sampled per mechanic. Ordered action ordinals are mandatory. Unknown, ambiguous, malformed, duplicate-ordinal, and missing-ordinal inputs must fail closed.

## Gates

All supported clauses, canonical graphs, action commands, and action sequences must compile exactly, and every safety challenge must fail closed. At 32 trials per intervention, language-condition schema recovery must be at least 0.90, mean target posterior at least 0.85, probability MAE at most 0.05, mean joint-distribution TV at most 0.05, and calibration error at most 0.05.

The language condition may degrade mean TV or held-out log loss by at most 0.01 relative to its paired symbolic baseline. Uniformizing outcome mass, shuffling action order, and literal language lookup must remain inadequate.

## Decision

A complete pass authorizes preregistration only of passive partial observation under stochastic dynamics. A language-only failure repairs the declared compiler. A shared symbolic and language failure returns to stochastic identifiability. V48 does not authorize active experiment selection, open ontology learning, model access, or neural training.
