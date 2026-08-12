# V19 preregistration: frozen grounding × executable schema induction

## Objective

V19 composes the frozen V15 language-grounding architecture with the unchanged V18 Boolean DSL,
version-space search, executor, latent episodes, support assignments, visible transition codes,
and queries. It isolates grounding errors from schema recovery; it does not train an adapter,
refit a linear head, select another representation, or construct a final mechanic.

V18 remains an oracle-grounded ceiling. V19 is development-only and cannot turn V17 into a fresh
holdout.

## Two language views

The primary `supported` view maps each episode's four latent variable positions to four determinant
concepts already present in V15. Known values use V15's affirmative current-observation surface,
and unresolved values use its established `UNKNOWN_CURRENT` construction.

The `novel_ontology` view keeps the identical latent episode, trace order, outcomes, and query
groundings while replacing those concepts with V18's fresh industrial, astronomical, archival,
or maritime ontology. It is diagnostic and cannot fail the primary gate.

Both views use the same candidate-action string and evidence structure. Thus their only intended
difference is determinant ontology. State-hypothesis pairs define the two Boolean values without
revealing which value is currently true.

## Locked two-by-two conditions

| Condition | Support grounding | Query grounding | Isolates |
|---|---|---|---|
| `oracle_support_oracle_query` | oracle | oracle | V18 ceiling |
| `frozen_support_oracle_query` | frozen V15 | oracle | support errors and schema recovery |
| `oracle_support_frozen_query` | oracle | frozen V15 | query execution and unknown handling |
| `frozen_support_frozen_query` | frozen V15 | frozen V15 | complete frozen neuro-symbolic system |

The primary unit is the episode. Query-pooled metrics remain secondary.

## Frozen grounding interface

Each support or query scene is serialized into the exact V15 prompt contract:

1. four listed Boolean determinants;
2. two ordered current-state hypotheses per determinant (`active`, then `inactive`);
3. four exact evidence units with character spans;
4. one matched unit per determinant;
5. `CURRENT` plus entailed/contradicted polarity for known values; or
6. `UNKNOWN_CURRENT` plus two unknown relations for unresolved values.

The Qwen3.5-4B revision, layer 8 representation, system prompts, evidence-span pooling,
hypothesis-token pooling, and three deployment heads are immutable. The historical deployment-head
artifact is eligible only if a pre-extraction audit reproduces it exactly from V15 development
features and confirms that it was fit without V17 examples or labels.

## Inconsistent support policy

After every frozen-grounded support trace, V19 reports whether the oracle target behavior remains
in the version space and how many behavioral candidates remain. If the version space becomes
empty, V19:

- records an explicit empty-version failure;
- returns the complete visible outcome vocabulary for every downstream query;
- returns `identifiable = false`;
- does not discard or repair a support;
- does not inspect the oracle grounding to choose a subset; and
- does not expand the DSL.

No confidence-aware repair is permitted in V19.

## Metrics

Grounding metrics:

- determinant allowed-value accuracy;
- exact scene grounding;
- exact support assignment;
- exact query grounding across `active`, `inactive`, and `unresolved`;
- fraction of episodes with every support grounded exactly; and
- results by supported versus novel-ontology view.

Schema metrics:

- target-behavior retention after every support prefix;
- empty-version-space rate;
- remaining behavioral hypotheses;
- unique target recovery versus target merely retained; and
- recovery conditioned on zero, one, or multiple support-grounding errors.

End-to-end metrics:

- episode-macro possible-transition-set exact match;
- complete-episode accuracy and worst episode;
- balanced identifiability accuracy;
- outcome-sensitive and outcome-invariant unknown accuracy; and
- the four-condition decomposition by development axis and language view.

## Pre-extraction gate

Feature extraction is forbidden unless all of the following pass:

1. V18 strengthened baselines and semantic/firewall audits pass;
2. both V19 views preserve every latent item and oracle grounding exactly;
3. the supported view uses only registered V15 concepts and temporal/operator constructions;
4. the novel ontology is isolated as non-gating;
5. agent inputs contain no allowed values, assignments, transition table, or executable program;
6. all prompt spans tokenize without truncation at 512 tokens;
7. prompt deduplication limits new model forward passes to at most 1,000;
8. the deployment heads reproduce exactly from locked V15 features; and
9. zero adapter runs, new head fits for selection, V17 record reads, V17 score reads, or final
   mechanic constructions occur.

If a gate fails, extraction remains forbidden and V19 must revise the interface rather than tune
the model.

## Decision rule

The supported fully frozen condition passes development if episode-macro transition-set exact
match is at least 0.50, the empty-version rate is at most 0.25, and the target behavior remains
after all supports in at least 0.75 of episodes. These thresholds are fixed before V19 features
exist. The two partial conditions determine whether support grounding or query grounding is the
dominant failure mode.

Passing V19 authorizes design—not access—of a fresh multi-mechanic final suite. It does not
authorize LoRA. Failure selects a grounding-specific or consistency-aware next experiment from
the preregistered decomposition.
