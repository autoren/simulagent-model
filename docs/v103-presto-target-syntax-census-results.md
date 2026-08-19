# V103 PRESTO Target-Syntax Census Result

## Outcome

V103 completed the frozen text-free diagnostic and closed PRESTO as the paired insufficiency source.
Across 12,279 en-US human-context dev/test records, all target strings used parentheses. The principal
literal form was the README-documented double guillemet:

| Diagnostic stage | Guillemet records |
|---|---:|
| Delimiter present | 3,246 |
| Quality-filtered literal | 3,226 |
| Literal present in admissible context | 2,015 |
| Literal absent from current input | 0 |
| Absent from input and present in context | 0 |

ASCII double quotes appeared in two records, with one quality literal and one context match, but again
zero literals absent from the current input. Single guillemets, curly double quotes, ASCII single quotes,
and square brackets supplied no quality literal candidates.

## Interpretation

V102r1's zero count was not caused by using the wrong principal delimiter. PRESTO contains target
literals and many repeat values from context, but every quality literal in these en-US human-context
records is also present in the current utterance. Therefore the source cannot instantiate the exact
independently checkable intervention required here: preserve the same current utterance and target while
making a required literal available only through prior context.

This does not imply that PRESTO context is useless. Context may affect nonliteral structure, reference,
or discourse phenomena that this exact-match diagnostic cannot certify. Those alternatives would require
a different ground-truth relation and are outside the preregistered construction.

## Boundary and decision

No input, target, literal, context, token, candidate identifier, or semantic root name was emitted or
manually inspected. There were zero model loads, generations, API calls, training runs, service calls,
and external side effects.

Freeze V103 negative and close the PRESTO paired-insufficiency branch. Do not revise delimiters, select
a PRESTO population, or infer LLM abstention performance. Continue the main study with the already viable,
balanced MASSIVE open-set population. If an evidence-ablation control remains scientifically necessary,
construct it prospectively as an explicit controlled intervention and label it as such—not as naturally
occurring PRESTO evidence.
