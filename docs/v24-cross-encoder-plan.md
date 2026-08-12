# V24 protocol: candidate-conditioned frozen relational cross-encoder

## Objective and status

V24 tests whether V22r2a failed because separately encoded evidence and atom spans are an inadequate
interface for exact entity and relation comparison. It is an exposed-data development experiment:
all V22r2 splits, predictions, and V23 results are known. V24 cannot support a holdout or final
claim, and it creates no fresh benchmark records.

The typed ontology, candidate atoms, entity aliases, V22 program catalog, executor, support outcomes,
Qwen revision, extraction layer, and model weights remain fixed. No LoRA or grammar change is
permitted.

## Recall-preserving proposal stage

For each evidence clause, the immutable V22r2a matcher supplies its three highest-scoring candidate
facts. The candidate selected by V22r2a's global one-to-one assignment is also included. Duplicate
edges are removed, leaving three or four candidates per evidence. Including every hard-assignment
edge guarantees that the sparse proposal graph admits at least one perfect matching.

The proposal stage is not the V24 prediction. It only bounds cross-encoder cost. Gold proposal
coverage is reported by split and role because the corpus is exposed. Evaluation support and query
coverage must each be at least 0.95 before model access.

## Candidate-conditioned representation

Each proposed pair is independently rendered as typed entities, action binding, one evidence
statement, and one candidate fact, in that order. The candidate fact therefore occurs after the
evidence in the causal prompt. Layer-8 candidate-fact tokens are mean-pooled in float32 under the
fixed system instruction to preserve exact entity identity, directed argument order, and
true/false/unresolved status.

The prompt contains no atom key, same-atom label, truth label, axis, target program, transition code,
or query answer.

## Fixed heads and assembly

Two heads are fitted once on `grounding_fit`:

1. a balanced C=1 binary `liblinear` head predicts whether the evidence concerns the candidate;
2. an explicit one-vs-rest set of balanced C=1 `liblinear` heads predicts
   `false`, `true`, or `unknown`, using positive candidate pairs only.

Both estimators use the fixed random state 2401.

No calibration or evaluation result selects a feature, C, threshold, proposal count, solver, or
branch policy. At inference, cross-match probabilities fill only registered proposal edges and a
maximum-weight one-to-one assignment creates the atom graph. Truth is read from the assigned pair.

## Evaluation and decision

Grounding metrics match V22r2a and include atom assignment, ordered relations, truth status, exact
scene/support/query graphs, semantic operators, and surface banks. The unchanged V22 inducer then
runs oracle/oracle, frozen/oracle, oracle/frozen, and frozen/frozen conditions with target-retention
and empty-version diagnostics.

Passing every registered development gate authorizes freezing this interface before construction of
a genuinely fresh surface benchmark. Failure localizes the next change to proposal recall,
candidate comparison, truth semantics, or symbolic integration. No V24 result directly authorizes
a final claim, LoRA, a neural challenger, or DSL expansion.
