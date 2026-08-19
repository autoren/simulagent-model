# V139 Repaired Direct-versus-Thinking Realization Results

## Outcome

V139 completed the exact locked recovery comparison: 200 deterministic local generations, one model load,
no retries, no raw-response or reasoning-trace persistence, no V134 or external-language access, no API,
and zero execution. The automatic decision was:

> `neither_condition_realizes_controlled_boundary_close_local_recovery_branch`

Neither condition cleared every noncompensatory gate. Unlike V137, however, the repaired thinking condition
produced valid and scientifically interpretable evidence.

## Direct condition

Direct inference was structurally valid on all 100 fixtures and perfect whenever decisive information was
present, but it guessed too often when the request was underdetermined:

- overall exact accuracy: 94%;
- clear accuracy: 100%;
- clarification-resolved accuracy: 100%;
- ambiguous abstention: 70%;
- full five-stage group accuracy: 70%;
- sequential query rate: 70%;
- sequential mean cost: 1.16;
- false-known rate on the difficult right branch: 10%;
- safe non-known rate on that branch: 90%.

All six errors were ambiguous cases. Fourteen of twenty received `A00`; the others were split between
`K01`, `N03`, and `U00`.

## Thinking condition

Thinking materially improved the controlled decision policy:

- overall exact accuracy: 97%;
- clear accuracy: 100%;
- clarification-resolved accuracy: 97.5%;
- ambiguous abstention: 90%;
- full five-stage group accuracy: 85%;
- sequential query rate: 90%;
- sequential mean cost: 0.62;
- false-known rate on the difficult right branch: 5%;
- safe non-known rate on that branch: 95%.

The sequential policy therefore passed its cost, improvement, false-known, and safe-non-known gates. Its
mean cost was 0.54 below direct and 0.73 below its own no-query behavior.

The paired comparison is also favorable: 93 fixtures were correct in both regimes, thinking repaired four
direct errors, direct was correct on one fixture that thinking missed, and both missed two ambiguous
fixtures. The gain is therefore not just an aggregate reshuffling.

## Why thinking still did not qualify

Only two gate families failed:

1. Ambiguous abstention was 90%, below the frozen 95% requirement. Two valid ambiguous outputs still
   overcommitted—one to `K01` and one to `N03`.
2. Structural validity was 97%, below the frozen 99% requirement. All three invalid outputs used exactly
   1,024 tokens and failed to close the prompt-opened thinking trace. Two were ambiguous and safely mapped
   to `A00`; one was a clarification-resolved `K04` case and became an erroneous abstention.

Because invalid output maps safely to `A00`, the headline 90% ambiguous accuracy includes two invalid
fallbacks. Restricted to valid ambiguous outputs, thinking correctly abstained on 16 of 18 (88.89%). This
is still materially above direct's 14 of 20, but it confirms a genuine semantic overcommitment gap in
addition to the output-completion problem.

Thinking averaged 249.03 generated tokens versus 12.20 for direct and used 2,309.29 versus 675.06 generation
seconds. It was about 3.42 times slower and generated about 20.4 times as many tokens. Total wall time was
2,986.63 seconds and peak active memory was about 16.7 GB.

## Interpretation and boundary

V139 establishes that thinking can improve abstention and downstream decision cost on this controlled
minimal-pair task. It does not show that thinking is sufficient: the local model still sometimes converts
semantic resemblance into membership and sometimes fails to finish within a large reasoning budget.

Freeze V139 as a positive paired mechanism result but a negative qualification result. Do not rerun,
increase the ceiling again on this population, inspect individual traces, or open V134. The correct next
step is an aggregate, model-free qualification-gap audit that separates safe completion handling from
semantic evidence sufficiency and defines a fresh successor without tuning on these fixtures. External
language, APIs, training, induction, authority, action, and execution remain closed.
