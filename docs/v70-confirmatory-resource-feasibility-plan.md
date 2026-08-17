# V70 resource-only feasibility preflight

This stage uses only the already-locked source inventory dimensions and V70 planner dimensions. It computes conservative upper bounds for the complete depth-0/1 census, Bellman branching, dense convergence transition tensors, and joint beliefs. It does not construct a policy, evaluate a reward, compute EIG or regret, generate confirmatory histories, or rescore development data.

All thresholds are frozen before running the calculation. If any threshold fails, the entire nine-model confirmation is deferred. No model may be removed, replaced, or assigned a smaller scientific role.
