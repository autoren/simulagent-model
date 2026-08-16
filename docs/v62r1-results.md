# V62r1 results: terminal Bellman-residual measurement repair

Repair qualification: **PASS**

V62 remains an immutable 31-of-32-gate failure. V62r1 rescored the same 66 reachable belief nodes with the preregistered terminal-aware independent checker; it did not repeat candidate evaluation or any of the 24 external rollout cells.

All four old residual failures were terminal-support beliefs, and there were no nonterminal old failures. The corrected maximum residual was `1.7763568394e-15` overall, `0` on terminal nodes, and `1.7763568394e-15` on nonterminal nodes. All other 31 V62 gates and every exact and official-rollout record reproduced without change. All repair fixtures passed and all six targeted mutants were killed.

The combined V62/V62r1 evidence therefore supports only exact finite-state, finite-horizon Bayesian filtering and planning transfer on the three pinned POBAX models. V62r1 is a measurement correction over immutable artifacts, not an independent external replication. It does not establish SMC2 portability, unknown-program inference, generic POMDP scalability, continuous or long-horizon control, formal safety, human-authored language robustness, or model/adapter performance. V58 remains deferred.
