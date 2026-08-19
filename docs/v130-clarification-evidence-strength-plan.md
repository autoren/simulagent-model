# V130 Clarification Evidence-strength Audit Plan

## Question

V129 showed that a complete typed answer is identifying but one 95%-correct answer is too weak against a
strong wrong-candidate prior. V130 derives how much evidence the frozen loss geometry actually requires.

## Prospective calculations

Use the same abstract 66-pair census, eleven-answer channel, three priors, and three error-bias regimes.
For one answer, evaluate the fixed 101-point reliability grid from 0.9500 through 1.0000 in increments of
0.0005 and record the first point passing all absolute regret, known, unsupported, and false-known gates.

Separately test one, two, and three 95%-correct answers. Charge 0.30 per answer. Dependence is a mixture of
independent draws and a common shock that repeats one draw across every answer, at correlations 0, 0.25,
and 0.50. The planner knows the correlation. This deliberately prevents repeated samples from being treated
as independent by construction.

## Feasibility decision

A single-source route is feasible only if every prior/bias condition passes at reliability no greater than
0.99. A multi-source route is feasible only if one answer count no greater than two passes every prior and
bias through correlation 0.25. Either result authorizes only a separately locked realization audit.

No language, benchmark record, human, model, protected data, capability induction, richer planning, API,
training, authority, or execution is permitted.
