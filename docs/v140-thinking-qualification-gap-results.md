# V140 Thinking Qualification-Gap Audit Results

## Result

V140 confirms that V139's near-miss contains two independent failure mechanisms.

The structural gate required at least 99 valid outputs. Thinking produced 97, so at least two fewer invalid
outputs are required. All three invalid outputs reached exactly the frozen 1,024-token ceiling and left the
prompt-opened reasoning trace unclosed.

The ambiguity gate required at least 19 correct abstentions among twenty cases. The safe fallback policy
produced eighteen, so at least one additional ambiguous correction is required. More strictly, two of those
eighteen were invalid outputs mapped to `A00`; among valid outputs, the model abstained correctly on sixteen
of eighteen and overcommitted twice.

Making the three existing fallback decisions structurally valid without changing their choice IDs would
pass validity but leave ambiguous accuracy at 90%. Correcting the two valid semantic overcommitments while
leaving completion unchanged would pass ambiguity but leave validity at 97%. Neither single-mechanism
counterfactual qualifies.

The paired gain remains real: thinking repaired four direct errors, introduced one, and left two ambiguous
errors unresolved, for a net three-fixture improvement.

## Decision

Freeze the result as a two-mechanism qualification gap. A justified successor must combine:

1. bounded finalization that produces a valid small answer without simply raising the unbounded reasoning
   ceiling or adding post-hoc retries; and
2. an explicit evidence-sufficiency mechanism aimed at resemblance-driven overcommitment.

These components must first receive a model-free feasibility protocol and then, if justified, be evaluated
on a fresh population. Multiple passes from the same model cannot be called independent evidence. V139 and
V134 must not be rerun or opened. External language, APIs, training, induction, authority, action, and
execution remain closed.
