# V63r1 repeat-pooling measurement-repair results

**Qualification:** PASS
**Decision:** `authorize_preregistration_of_separate_multi_action_external_EIG_stage`

V63r1 restored the V53r2 aggregation rule that V63 had intended to transfer: form an equal-weight
posterior mixture of the three independent SMC² repeats, then compute one metric row per record and
particle budget. Original V63 remains failed. V63r1 is a measurement repair, not independent
replication.

## Primary pooled exact-reference agreement

| Metric | Mean | q95 | Gate |
|---|---:|---:|---:|
| Identity TV | 0.000371647 | 0.00193997 | 0.03 / 0.10 |
| Theta Wasserstein | 0.00213659 | 0.00413164 | 0.03 / 0.10 |
| Binned joint identity-theta TV | 0.0443754 | 0.0624986 | 0.06 / 0.18 |
| Current-side TV | 0.00107353 | 0.00365769 | 0.05 / 0.15 |
| Next-observation TV | 0.000939161 | 0.00256312 | 0.05 / 0.15 |

Mean absolute log-evidence error was `0.0174005`. Composite mean error decreased monotonically from
`0.0422910` at 31 outer particles to `0.0217206` at 127 and `0.0118804` at 509. All seven controls
were detected or dominated, normalization was exact, target identity never became extinct, false
collapse rates were zero, and stream/fingerprint collision rates were zero.

## Reused sealed evidence

V63r1 did not rerun SBC, scale, or the external runtime. Their original V63 subsections and runtime
artifact were reused by hash:

- SBC minimum chi-square p-value `0.00845262`, maximum rank-bin z `3.25`, maximum coverage z
  `2.35151`;
- scale completion and normalization `1.0`, with zero target-identity extinction;
- pinned POBAX transition and observation array error `0.0`, maximum empirical probability error
  `0.0134375`.

All noncompensatory gates passed. This qualifies only calibrated SMC² portability for one
project-authored unknown-dynamics layer anchored to the pinned five-state POBAX Tiger model. POBAX
does not itself supply that unknown-dynamics family.

The next authorized action is preregistration—not execution—of a separate external model with at
least two informative nonterminal interventions for EIG versus fixed/random comparison. Tiger's
single informative `listen` action remains ineligible as the substantive active-design test.
Reward planning, policy verification, safety claims, V58 human-language work, model access, and
adapter training remain unauthorized.
