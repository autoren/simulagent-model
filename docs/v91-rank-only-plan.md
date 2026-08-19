# V91 Rank-Only Structured-LLM Plan

## Question

V90 ruled out parameter count, model generation, and 8-bit precision as sufficient remedies for using a
local model to generate the authoritative semantic candidate set. V91 asks a narrower question: can the
low-cost frozen Qwen3.5 4B model usefully prioritize an independently complete intent list without being
able to remove an intent, alter accumulated state, assign a posterior, choose an action, or execute a
tool?

## Safety-preserving interface

Deterministic code reads the frozen service schema and creates the complete list of every service intent
plus the operational `NONE` state. The model may return only a proposed `intent_priority` list. A
fail-closed canonicalizer keeps the first occurrence of each allowed identifier and appends every omitted
identifier in frozen schema order, including `NONE`. Malformed output is equivalent to an empty proposal,
so it still produces the complete deterministic order.

The model therefore controls only scheduling. It cannot prune the search space. Full enumeration must
produce the identical hypothesis set, posterior, policy value, and action regardless of model output.
No early stopping, approximate posterior, direct action, state-key generation, belief update, service
call, or side effect is part of V91.

## Fresh-source rule

Before any payload access, V91 pins `dev/dialogues_003.json` from the official Schema-Guided Dialogue
repository at commit `e852981ae34990f4358979625854259302feaa78` by byte size and Git blob identity.
The source stage may emit only a text-free structural inventory. Population strata, controls, prompts,
decoding, scoring, gates, and the policy-invariance harness must be frozen before selected utterances are
extracted or the model is loaded.

## Intended controls and measurements

The later preregistration may compare the local ordering against frozen schema order, deterministic
lexical retrieval, a bounded exact-match grammar, exhaustive/unordered enumeration, and an oracle-first
ordering. Primary utility measurements will be gold rank, top-k recall, reciprocal rank, and work to
first gold. Safety measurements will require exact complete-set equality, mandatory `NONE` retention,
zero state mutation, and exact downstream posterior/value/action invariance under complete enumeration.

The model condition may qualify only if every safety gate passes and it materially improves ranking over
the best non-oracle deterministic control. A pass could authorize a separately preregistered bounded
search-scheduling study only. It cannot authorize pruning, learned likelihoods, API access, adapter
training, belief authority, action authority, deployment, or execution.
