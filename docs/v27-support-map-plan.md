# V27 protocol: outcome-constrained support MAP

## Objective and status

V27 targets the remaining support-stage failure without changing successful query inference. It is
an exposed-data development experiment. V26 query predictions are copied unchanged. The model,
V24 proposal graph and match head, V26 decoder, ontology, DSL, executor, and observed support
interface remain fixed.

The V26 decomposition found that oracle truth over the fixed V24 assignments would yield 0.942
frozen-support/oracle-query execution, while actual V26 support yields 0.545. Decoder margins are
also much lower on incorrect support atoms (median 0.82 versus 3.03). This supports joint support
selection under the already-public observed transition codes.

## New model work

V24 proposed 1,652 support edges. V26 already scored the 549 selected edges. V27 reuses those scores
and performs exactly 1,103 new full-depth native-decoder forwards for the remaining edges. The
prompt, A/B/C labels, float32 direct projection, model revision, and precision are identical to
V26. No head or threshold is fitted.

## Fixed joint objective

For each support scene, every sparse perfect matching is enumerated; the pre-model audit requires at
most 256. A selected edge contributes the V24 binary match log-odds. Its truth label contributes the
log-softmax probability from V26-style A/B/C logits. Both factors have fixed weight one.

For every assignment, the exact top 64 independent truth vectors are generated. After excluding
graphs with more than four unknown atoms, the top 512 joint graphs are retained. For each episode,
one program and one graph per support scene are selected to maximize summed graph log score subject
to every graph allowing its public observed transition under that shared program. Program ties use
program key; graph ties use canonical graph key. If no feasible program exists, the V26 support
graphs are retained.

## Decision

V27 passes only if evaluation support exact graphs, target retention, empty version spaces, and all
registered execution conditions clear their gates. A pass authorizes a separate query/exact-graph
repair stage; it does not yet authorize a fresh benchmark because V26 full-scene exactness remained
below its declared gate. Failure rejects outcome-constrained MAP at these fixed branch budgets and
does not authorize LoRA or post-hoc weight/branch tuning.
