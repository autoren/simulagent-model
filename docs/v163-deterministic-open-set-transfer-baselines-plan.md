# V163 deterministic open-set transfer baselines plan

## Purpose

V163 evaluates deterministic controls on the V162 development-transfer artifact while keeping protected
language sealed. It reuses the frozen V105 training-only visible catalog and safe 17-hypothesis universe for
comparability with the earlier MASSIVE sequence. No selected V162 utterance was read while designing this
stage.

This is a controlled open-set capability benchmark. It does not test V160's relation-alias codebook.

## Split and controls

Hash-split each of the four 48-record classes into 24 calibration and 24 evaluation records using frozen
identifiers and class labels only. Character retrieval trains only on MASSIVE train utterances for the 12
declared intents and tunes its two thresholds on the 96 calibration records. The evaluation set is touched
once after the rule, threshold grid, costs, metrics, and gates are locked.

Controls are complete safe enumeration, ask always, the unchanged identifier grammar, calibrated character
n-gram retrieval, deterministic consensus, and the hidden oracle. Twenty-four identifier-selected
missing-observation controls expose no utterance and must abstain.

## Consensus and residual

Deterministic consensus accepts a decision only when identifier grammar and character retrieval produce the
same complete non-abstaining status and arguments. Every disagreement maps to `ABSTAIN`. The residual is
exactly the evaluation identifiers on which this rule abstains. Membership uses predictions only; truth and
language cannot affect it.

A later local-model protocol is eligible for preregistration only if:

- the residual is nontrivial but not effectively the whole population;
- at least eight records are handled without a model;
- nonresidual exactness is at least 95%;
- nonresidual false-known acceptance is zero;
- nonresidual mean regret is at most 0.25;
- consensus overall false-known acceptance is at most 5%; and
- consensus is no more costly than ask always.

These gates test whether deterministic evidence safely removes an easy subset. They cannot be relaxed or
retuned after the run. Failure closes the local-model residual branch on this population.

## Boundary

V163 may automatically read development language and declared training language once after its design lock.
It may not open protected language, inspect an utterance manually, load a model, use an API, train, induce an
ontology, alter authoritative state, act, execute, call a service, or create a side effect.
