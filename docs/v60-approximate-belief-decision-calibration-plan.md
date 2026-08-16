# V60 approximate-belief decision-calibration plan

V59 established that bounded root-sampled observation-contingent search works when initialized from the exact joint belief. V60 changes only that input: it replaces the exact belief with the already validated V53r2 SMC² approximation and asks whether posterior error remains small in decision space.

The 24 sealed V59 public tasks are reused, paired by construction. For each task, SMC² runs at 31, 127, and 509 outer particles with three independent repeats pooled equally. The exact 257-node quadrature posterior remains the reference. Both beliefs feed the same 1,024-simulation root-sampled search under common seeds; all deployed policy returns are evaluated independently under the exact posterior.

At horizon 3, exact dynamic programming measures root-action optimal-set membership and exact root regret. At horizons 5 and 7, the primary endpoint is return loss relative to exact-belief search, with an equal-budget observation-blind control checking that approximate beliefs still support useful observation contingency. Posterior agreement, decision calibration, return preservation, budget accounting, replay, and all access firewalls are noncompensatory.

No new population is constructed. Candidate code may open only the V59 sealed public artifact. The audit-truth artifact remains forbidden. A pass would qualify only the frozen SMC²-to-budgeted-search composition; it would not establish exact long-horizon optimality, general-purpose inference, formal safety, or human-language robustness.
