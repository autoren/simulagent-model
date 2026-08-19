# V167r1 history-action metric repair plan

V167r1 preserves the frozen one-shot V167 run and corrects one summary projection before any nominal outcome is
created. The original code treated `STOP/defer` and `STOP/provisional` as different second actions because it
included the terminal decision in the comparison. The registered question concerns the second action itself, so
both are `STOP`; a query action remains identified by its valuation index.

V167r1 independently reconstructs all V167 cases and risks, projects child actions from the persisted exact policy
trees, requires the original value 48 and corrected value 28, recomputes the one affected gate, and verifies that
all other metrics, gates, risks, artifacts, and the branch decision remain exact. It does not modify or rerun V167.

No evaluation data, judgment, model, API, training, registration, trusted-state mutation, service, side effect,
action, or execution is allowed. Passing authorizes only a separately locked reversible-sandbox design.
