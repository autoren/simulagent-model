# V202 model-free controller decision-sufficiency results

## Result

V202 is a positive development result. `SINGLE_PRESENTATION_TOP3_FAMILY` passed every frozen cost, robustness,
incremental-value, and safety gate and was selected by the preregistered one-call-first rule.

The selected family asks one trusted multiple-choice question containing the model's top three contracts plus
`OTHER`. It does not treat ranks two and three as posterior probabilities, discard the full authoritative
hypothesis universe, or permit the model to decide or act.

## Fixed-policy comparison

| Policy | Calls | Robust primary cost | Robust macro cost | Worst improvement over matched `CHAR_LAST` | Target-hit disagreement | Qualified |
|---|---:|---:|---:|---:|---:|---|
| `SINGLE_PRESENTATION_TOP1_FAMILY` | 1 | 0.14792 | 0.14167 | 0.11097 | 0.10714 | No |
| `SINGLE_PRESENTATION_TOP3_FAMILY` | 1 | 0.21417 | 0.21190 | 0.04417 | 0.01190 | Yes |
| `TOP1_PLURALITY_3X` | 3 | 0.14403 | 0.13333 | 0.11486 | 0.00000 | Yes |
| `TOP3_INCLUSION_CONSENSUS_3X` | 3 | 0.21417 | 0.21190 | 0.04417 | 0.00000 | Yes |

Top-1 narrowly failed only the frozen target-hit disagreement gate: presentation changed whether the true target
was present on 9 of 84 records (`0.10714`), above the `0.10` maximum. Its low mean cost therefore was not enough.

The three-call plurality policy had the lowest qualified cost, but it was ineligible once a one-call family
qualified. This prevents a small question-cost gain from concealing a threefold inference requirement. The top-3
consensus also qualified but did not improve robust cost over the one-call top-3 family.

For the selected top-3 family:

- robust primary mean cost was `0.21417`, below the `0.24` gate;
- robust macro mean cost was `0.21190`, below the `0.25` gate;
- worst-presentation primary improvement over matched `CHAR_LAST` was `0.04417`, above the `0.01` gate;
- target-hit disagreement was `0.01190` and mean per-record cost range was `0.00238`;
- mean proposal size was `2.83333`;
- target retention and exact completion after trusted answers were both `1.0`, with zero false terminal decisions.

## Interpretation

V201 showed that the identities of lower-ranked alternatives were presentation-sensitive. V202 shows that this
does not prevent those alternatives from being decision-sufficient for a bounded clarification question. The
scientifically defensible interface is therefore:

1. the LLM proposes a small candidate set;
2. a deterministic controller adds `OTHER` and asks a trusted source;
3. the full authoritative universe remains available;
4. only the trusted answer can produce an exact terminal decision.

This is weaker than calling the ranked output a calibrated posterior, but stronger and more useful than discarding
the model because its complete top-3 set is not invariant.

## Boundaries and next step

The evaluation read 252 normalized model fixtures and 252 matched deterministic predictions. It read no utterance
language or raw model responses, loaded or generated with no model, made no API or service calls, performed no
training or registration, mutated no trusted state, and executed no action.

This remains a development result on the V191 population. It does not reopen V198 protected language and does not
authorize immediate confirmation. The next evidential step is a separately preregistered fresh or external
confirmation of the fixed canonical top-3-plus-`OTHER` controller. If an adequate independent natural-language
population cannot be obtained without reusing protected material, this branch should remain development-only and
the main program should advance to the model-free richer POMDP track.
