# V167 exact evidence-gathering planner results

## Outcome

The V167 development planner passed every scientific gate after a verifier-only correction to one summary
projection. Exact Bayes-adaptive querying had lower expected total risk than immediate deferral, forced MAP,
random open-loop pairs, greedy class information gain, and the optimal fixed open-loop pair on all 48 frozen
ambiguous cases.

Mean expected total risk was:

| Policy | Mean risk |
|---|---:|
| Oracle class | 0.0000 |
| Exact Bayes-adaptive | 1.3829 |
| Greedy class information gain | 1.4060 |
| Optimal open-loop pair | 1.4060 |
| Random open-loop pair | 1.7856 |
| Immediate Bayes terminal decision | 2.0000 |
| Forced MAP without defer | 3.3333 |

All 48 cases had positive value of information. Exact adaptive planning strictly improved over the best fixed
query pair in all 48. Root queries varied across valuation indices 3, 5, and 7, so there was no globally dominant
query. After correcting the action projection, 28 cases had a first outcome that changed whether or which second
query was selected. All renaming-equivalent risks matched exactly, and all hidden target candidates remained in
their initial version spaces.

## Metric repair

The frozen V167 implementation originally reported 48 history-dependent second-action cases. It compared the
first two fields of child policy tuples, so `STOP/defer` and `STOP/provisional_primitive` were incorrectly counted
as different second actions. The intended action-level projection compares only `STOP` versus the specific
`QUERY_Vi`. The corrected count is 28. This remains above the preregistered minimum of one and changes no policy,
risk, other metric, gate, or branch decision.

The original one-shot run and artifacts remain immutable. V167r1 freezes this projection repair rather than
editing or rerunning V167.

## Interpretation

V166 showed how to retain uncertainty; V167 shows that retained uncertainty can support useful sequential
decisions. The information-gathering benefit is not merely entropy reduction. An adaptive policy sometimes stops
after one answer and sometimes asks a different second question, saving interaction cost while respecting the
high loss for false provisional creation.

The result is development-informed. Two feasibility calculations on project-authored hidden development truth
were disclosed before the formal lock and informed the prior, loss, cost, and horizon. It is mechanism evidence,
not fresh or external confirmation.

No model, API, training, ontology registration, trusted-state mutation, service call, side effect, real action,
or execution occurred. Candidate beliefs and terminal decisions remained shadow-only.

## Decision

Freeze corrected V167 through V167r1 and preregister Track B's fixed-ontology reversible sandbox under a separate
lock. Do not integrate provisional concepts or run a model on V166's empty residual.
