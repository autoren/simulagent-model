# V63 external unknown-dynamics inference plan

V62r1 measurement-repaired the external exact-belief transfer result without changing the fact
that original V62 failed its registered terminal-state residual. V63 takes the next authorized
step: test whether the already frozen V53r2 SMC² structure transfers to a small external POMDP
anchor. It does not rerun V62, access V58, simulate human language, or touch model weights.

## External anchor and added family

The anchor is `tiger-alt-start.POMDP` from POBAX commit
`a5e1d62d14e4efe783885b9d4f19cffa2a568eec`, already copied and hash-sealed by V62. POBAX supplies
the state, action, observation, reward, sensor, discount, and terminal semantics. It does **not**
supply an unknown-dynamics benchmark. V63 adds one explicitly project-authored uncertainty layer
and must retain that boundary in every result.

The hidden identity is `persistent` or `alternating`, with equal prior mass. The continuous
parameter is \(\theta\sim 0.65+0.30\operatorname{Beta}(2,2)\). On `listen`, persistent dynamics keep
the tiger side with probability \(\theta\), whereas alternating dynamics switch it with probability
\(\theta\). The remaining mass takes the opposite transition. Both start and post-listen side
states use this rule. Open actions, observations, rewards, discount, initial state, and terminal
semantics remain byte-derived from the pinned model.

This parameterization excludes \(\theta=0.5\). If side and sensor reports are encoded as \(\pm1\),
then the lag-one report correlation is

\[
  \operatorname{corr}(Y_t,Y_{t+1}) =
  s_M(2\cdot0.85-1)^2(2\theta-1),
\]

where \(s_M=+1\) for persistent and \(-1\) for alternating. Its sign identifies the discrete
identity and its magnitude identifies \(\theta\). On the registered support the two identity ranges
are disjoint: `[0.147, 0.441]` versus `[-0.441, -0.147]`, and the absolute slope in \(\theta\) is
`0.98`.

Before a population may be constructed, an independent design audit must bind the source hashes,
reproduce those analytic quantities, exhaust all binary report histories of lengths two through
six, and run an exact augmented-belief planning census at remaining horizons one through five. The
family is admissible only if observations move identity mass materially, exact planning uses both
listen and open actions, and at least one MAP/point-parameter collapse action has positive regret
under the exact joint posterior. This is a feasibility check over the declared model, not candidate
algorithm evaluation.

## Exact inference, SBC, and frozen SMC² transfer

The exact benchmark has 32 records, balanced across identities. Each record contains four
independent all-listen support episodes with lengths eight and twelve plus a designated current
episode prefix of length six or ten. The exact primary reference enumerates the five hidden states
and both identities while integrating \(\theta\) with 257-node Gauss–Legendre quadrature under the
scaled Beta prior. A separately written scalar log-domain forward algorithm and independently
constructed quadrature must agree before particles can be scored.

The SMC² adapter inherits V53r2's outer budgets `31/127/509`, inner budget `127`, three exact-record
repeats, ESS thresholds, systematic resampling, two PMMH rejuvenation steps, and fixed logit-space
proposal scale `0.4`. Identity is enumerated exactly. Each identity-specific outer particle owns a
hidden-state particle filter; identity evidence is obtained from its SMC² normalizing-constant
estimate. New code is allowed only for the external family transition, observation, and data
adapter.

The 256-replication prior-predictive SBC population tests identity, continuous parameter, current
side, and posterior-probability quantities with 127 posterior draws, 16 rank bins, and 50/80/95%
coverage. Fixed controls include MAP identity/point parameter, theta point mass, iid report
likelihood, squared likelihood, identity swap, disabled rejuvenation, and an intentional stream
collision. A scale population reaches sequence length 64. The unchanged pinned POBAX runtime must
also reproduce declared transition arrays and sampling frequencies without source mutation.

Every accuracy, calibration, degeneracy, stream, control, runtime, mutation, scale, access, and
normalization gate is noncompensatory. Thresholds are inherited where comparable from V53r2. The
design, implementation, populations, evaluation implementation, and one-shot outcome are frozen in
that order.

## What a pass would and would not authorize

A full pass would establish only calibrated SMC² portability for this source-anchored five-state
unknown-dynamics family. It would authorize preregistration of a **separate** external EIG stage
whose model has at least two informative nonterminal actions. Tiger cannot serve as a substantive
active-design benchmark because `listen` is its sole informative nonterminal action; comparing it
with immediately terminal opens would make EIG success nearly tautological.

V63 therefore does not authorize active selection, reward-bearing planning evaluation, policy
verification, safety claims, language work, model access, or adapter training. Those remain gated
on later stages and V58 remains deferred until real independent participants exist.
