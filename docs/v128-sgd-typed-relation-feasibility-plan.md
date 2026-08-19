# V128 Fresh SGD Typed-relation Feasibility Plan

## Question

V127 showed that even perfect slot-name sets are insufficient. V128 tests a richer structural unit before
building a parser: typed dialogue-act/slot relations, accumulated state-slot presence, and requested-slot
presence. Source annotations remain an unavailable oracle upper bound.

## Prospective design

Select 432 test records before archive access: 144 known, 144 novel-valid, and 144 unsupported. Exclude all
V125 and V127 evaluation identifiers. From all 4,881 locked V125 known training frames, form one union
support set for each of six declared known intents using only these tokens:

- `ACTION::<act>::<slot>` for non-intent USER actions;
- `STATE_SLOT::<slot>` for accumulated state keys;
- `REQUEST_SLOT::<slot>` for requested-slot names.

No utterance or slot value may be accessed. No frequencies, probabilities, classifiers, likelihoods,
thresholds, or candidate selection are permitted. An evaluation relation set is compatible with a known
intent iff it is nonempty and contained in that intent's frozen training support. Skip clarification only
when exactly one intent is compatible; otherwise query through the unchanged V119 simulated channel.

The inherited selectivity, safety, regret, complete-retention, and zero-execution gates are unchanged.
Passing authorizes only a separately preregistered parser-realization design. Failure closes this entire
annotation-signature family rather than motivating post-hoc token or threshold mining.
