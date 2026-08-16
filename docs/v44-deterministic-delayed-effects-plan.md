# V44 oracle deterministic delayed-effect foundation

## Why this is next

V42 established persistent state across ordered actions. V43/V43r1 established that declared state and action language can populate that reasoner exactly within the paired scope. The next capability should therefore add one genuinely new temporal axis while preserving deterministic, exact semantics: effects that occur after a fixed delay.

V44 is oracle-first and language-free. It does not combine delay with stochasticity, active experiment selection, open concepts, or a neural challenger.

## Tick and queue semantics

Each action advances one tick. At the start of a tick, events due at that tick are delivered. The current action is then evaluated on the resulting state; its immediate effects are applied and any delayed payloads are enqueued. The post-action state is observed. An event scheduled with delay one at tick `t` is therefore delivered before the action at tick `t+1`.

The explicit `wait` action advances time without a direct effect. Pending events are not flushed automatically when a sequence ends. Conditions are evaluated when an event is scheduled. Delayed payloads are limited to set-true, set-false, and toggle; delayed copy is deferred. Programs and generated cases forbid multiple effects targeting the same atom on the same delivery tick, avoiding an arbitrary conflict policy.

## Population

The development population contains 40 mechanics, ten from each of one-tick unary set/clear, two-tick relational toggle, state-conditional scheduling, and interleaved immediate/delayed effects. Sequences have three to six actions over two to five entities. There are 24 structurally fresh queries per mechanic and no support/query overlap.

Every mechanic has a delay-sensitive query. Every family includes paired cases with the same initial state and action multiset but different placement of `wait`; at least one registered pair must have a different final observation. This makes delivery time causally relevant rather than decorative.

## Controls and gates

The primary system performs exact version-space induction over the queued DSL. A collapsed-delay control applies every delayed payload immediately. An end-flush control withholds every delayed payload until sequence end. Literal lookup is the non-lifted control.

The queued system must be exact for target retention, schema recovery, every intermediate and final observation, every family and sequence length, and every wait-placement counterfactual. Collapsed delay and end flush must each remain at or below 0.850 final exactness; lookup must remain at or below 0.950.

## Decision

- Queued execution passes and both timing controls fail: preregister declared-language grounding for delayed mechanics.
- Queued execution fails: repair the event queue or delayed DSL before adding language or stochasticity.
- A timing control passes: redesign the benchmark so timing is causally necessary.
- Lookup passes: repair the structural split.

This is a non-final deterministic temporal foundation, not a claim of general world-model learning.
