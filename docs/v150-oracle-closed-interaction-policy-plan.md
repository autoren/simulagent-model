# V150 oracle closed-interaction policy plan

V150 establishes decision value before any LLM is evaluated. It uses only V149 development metadata and never reads evaluation language. Each of 24 development groups yields two latent-side episodes beginning from the ambiguous state, for 48 episodes total.

The exact planner compares no query, the one registered discriminating query, and five irrelevant registered queries. No query safely abstains at cost 1.0. The correct closed query costs 0.3 and resolves the side exactly. An irrelevant query produces no selection and remains `A00`, so it costs 1.3 rather than forcing a false answer.

Every episode is crossed with all seven possible LLM state proposals and seven query proposals (`NONE` plus six queries), producing 2,352 policy evaluations. These proposals are explicitly non-authoritative: the exact planner uses the complete hidden pair and registered query model, and the typed witness firewall determines the final state.

Gates require exact query selection and final state, complete proposal invariance and hypothesis retention, 0.3 or lower mean cost, at least 0.7 improvement over no query, one resolved episode per query, zero false-known results, exact fail-closed irrelevant queries, and zero language/model/API/training/execution access.

Passing authorizes only design of a prospective local proposal protocol on development language. It does not authorize an immediate model run or evaluation access.
