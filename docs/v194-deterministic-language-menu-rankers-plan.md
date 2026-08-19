# V194 deterministic language-to-menu ranker plan

## Purpose

V194 supplies fixed model-free language controls before any local model is loaded. It reads only the fresh V192
unprotected development artifact and ranks the 14 visible V193 menu options. Targets and the hidden option map are used
only after ranking for automatic scoring.

## Rankers

Four fixed configurations are evaluated without fitting or thresholds:

1. character 3/4/5-gram cosine on the final user utterance;
2. the same character view on all user utterances;
3. token cosine on all user utterances; and
4. reciprocal-rank fusion of the all-user character and token rankings with constant 60.

System utterances are excluded. Ties use ascending option ID. Every observed record emits exactly three distinct menu
options; every missing record emits `INSUFFICIENT`.

## Economics and controls

Each ranking is scored under both frozen V193 policies. A hit costs `0.10` for top-1 or `0.20` for top-3. A miss pays
that question cost plus `0.40` generic clarification. Exact terminal state always comes from the trusted answer.

Primary economics use the frozen V193/V190 contract prior, divided equally among the six fresh records for each
contract. Balanced macro metrics weight the 84 records equally. Fixed hierarchy, always generic, target-informed
oracles, and random-recall expectations are reported.

## Gates and decision

Integrity, missing handling, target retention, exactness, and zero authority are noncompensatory. To justify one local
model comparator, at least one deterministic ranker must achieve 30% top-3 recall under both primary and balanced
weights—above the 3/14 random expectation. Material deterministic value (`<=0.36` primary mean cost) is reported but is
not required; the model comparison is intended to test value beyond these controls.

A pass authorizes only preregistration of one bounded local-model shadow comparator. It does not authorize immediate
model execution, an API fallback, protected access, ontology registration or pruning, trusted mutation, action, or
execution.
