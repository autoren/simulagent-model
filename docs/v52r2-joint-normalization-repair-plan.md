# V52r2 preregistration: final-joint Decimal normalization repair

V52’s single sealed run passed every accuracy, convergence, calibration, degeneracy, correlation-integrity, extinction, scale-completion, and control-sensitivity gate. It failed only the exact normalization-rate gates: 0.7685546875 on SBC and 0.7083333333 on scale stress.

The failure is arithmetic bookkeeping, not particle inference. On the first failed SBC record, the frozen filter produced support-program residual `1e-100`, query-program residual zero at 100 digits, suffix residual zero, and final-joint residual `1.2e-29`. The final joint and configuration products were the only relevant operations performed outside the declared 100-digit Decimal context, so Python rounded them at its process-default 28-digit precision. The preregistered diagnostic correctly rejected that residual because its tolerance is `1e-80`.

V52r2 changes exactly one operation: final joint and configuration assembly is performed and normalized inside a 100-digit Decimal context. It retains the same registry, sealed populations, particle budgets, seeds, particle trajectories, resampling decisions, exact oracle, controls, metrics, gates, and claim boundary. A pre-evaluation implementation fixture must show that filtering log likelihoods and diagnostics are identical, that base-versus-repair TV is at most `1e-25`, and that every returned marginal satisfies the frozen normalization tolerance.

Only one repair evaluation is allowed. A full pass authorizes preregistration—not construction or evaluation—of continuous-parameter SMC² with an offline PMCMC reference. Active intervention selection, reward, planning, language, model access, and final evaluation remain blocked.
