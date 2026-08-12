# V23 protocol: exposed-data probabilistic support replay

## Purpose and status

V23 asks whether V22r2a failed because hard support decisions destroy the true relational program
even when the frozen scores still rank the correct graph components highly. It is an exposed-data
development replay. Every V22r2 split and its registered result have been inspected; V23 cannot be
reported as a holdout or final result.

The frozen 4B features, atom and truth heads, typed ontology, candidate statements, V22 program
catalog, executor, support outcomes, and query graphs are unchanged. V23 performs no model forward,
feature extraction, linear fit, calibration fit, adapter training, or grammar expansion.

## Finite graph distribution

For each support scene, the saved binary atom-matching logit is mapped through a sigmoid. Exact
k-best one-to-one evidence/candidate assignments are enumerated with deterministic Murty
partitioning. The saved one-vs-rest truth logits are mapped through sigmoids and normalized across
`false`, `true`, and `unknown`; exact k-best independent truth vectors are enumerated.

Assignment and truth-vector log scores are added. Duplicate graphs are merged, proposals with more
than four unknown atoms are discarded, and the highest-scoring `B` graphs are retained and
softmax-normalized. The complete registered curve uses `B ∈ {1, 4, 16, 64}`. No budget is selected
from its result.

## Program posterior and queries

For each program and support trace, likelihood is the retained probability mass of graphs on which
the observed transition is possible. Trace likelihoods are multiplied in log space under a uniform
prior. A zero-likelihood program is removed. Query predictions use the union over the smallest
deterministically ordered credible program set reaching mass `m`, for
`m ∈ {0.50, 0.80, 0.95, 1.00}`.

Oracle query graphs are primary so V23 isolates support uncertainty. The registered V22r2a hard
query ceiling remains separately reported; V23 cannot claim complete language grounding.

## Anti-widening metrics and decision

Every curve cell reports target retention in the nonzero posterior and credible set, empty
posteriors, credible-program count, exact answer sets, complete episodes, predicted set size, excess
outcomes, missing target outcomes, proposal concentration, and runtime. Broader answer sets do not
count as a repair.

The reference point is fixed before execution at `B=64`, `m=0.95`. It authorizes construction of a
fresh benchmark protocol only if target nonzero and credible retention are at least 0.75, empty
posteriors at most 0.10, transition-set exact at least 0.75, mean excess outcomes at most 0.25, and
missing-target-outcome rate at most 0.05. Failure keeps the result diagnostic and calls for a
different language interface. No V23 outcome directly authorizes a final claim, LoRA, or a neural
challenger.
