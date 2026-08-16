# V65r1 pooled SMC² EIG portability outcome

Date frozen: 2026-08-16

## Bottom line

V65r1 did not complete and does not authorize Bayes-adaptive reward planning. Its sole immutable
evaluation attempt terminated with `RuntimeError: all V65 outer particles became extinct` before
the frozen evaluator wrote a result or raw record-budget cells. The one-shot authorization is
consumed; V65r1 will not be rerun.

This is an implementation-domain failure, not evidence that the approximate posterior passed or
failed the registered accuracy gates. Only the completion gate is known to fail. Every posterior,
predictive, EIG, selection, regret, and scaling gate remains unevaluable.

## Frozen pre-run evidence

Before the attempt, the evaluator and every runtime dependency were bound by hash and pushed to
`main`. The valid synthetic fixture passed, all 32 registered evaluator mutants were rejected, all
six analytic fixtures passed, the 48-record public subset was sealed, and no V64 audit, result,
truth field, human record, model forward pass, or adapter-training run was accessed.

Those checks remain useful, but they did not include a public history that is impossible under one
of the two identity-conditioned models while remaining possible under the other.

## Independently reproduced cause

After the process terminated, an exact support-only audit loaded the 48 sealed public histories. It
computed no candidate EIG values and did not rerun V65r1. Exactly one history has an identity branch
with zero likelihood:

- record: `c55d371eada6c66063aa84e9`;
- prefix length: 5;
- identity 0 log evidence: `-6.528606950551039`;
- identity 1 log evidence: `-Infinity`;
- identity 1 first becomes impossible at zero-based transition tick 4.

The frozen implementation runs SMC² separately for each identity. At line 526 of
`python/v65_smc2_eig.py`, it raises whenever all theta particles within either identity branch have
zero incremental likelihood. That behavior is incorrect for this case: a zero-likelihood identity
is a valid Bayesian result when another identity remains possible. The joint posterior should give
the impossible identity zero mass and continue.

The repair must still distinguish this exact structural zero from accidental finite-particle
collapse in a branch whose exact likelihood is positive. Silently treating every particle
extinction as model exclusion would create a new bias.

## Next authorized direction

V65r2 may be preregistered as a narrow extinct-identity repair. It must:

1. preserve V65r1, its failed attempt, the 48-record subset, budgets `[31, 127, 509]`, three-repeat
   pooling, inner budget 127, exact reference, controls, and every original accuracy gate;
2. represent an exactly impossible identity with log evidence `-Infinity`, zero joint posterior
   mass, no scored atoms, and complete work diagnostics instead of aborting the joint inference;
3. retain a hard failure for unexplained particle extinction when an independently computed exact
   finite-state support check says the identity has positive likelihood;
4. add pre-evaluation fixtures for one-identity extinction, both-identity impossibility, and
   finite-particle collapse under positive exact support;
5. make the evaluator persist a terminal failure artifact if any future exception occurs, without
   authorizing a retry; and
6. describe any later pass as a targeted V65r2 repair result, not an independent replication.

No reward planning, formal verification, human-data substitution, model access, or adapter training
is authorized by this outcome.
