# V137 Direct-versus-thinking Realization Results

## Outcome

V137 completed the exact locked comparison: 200 deterministic generations, one model load, no retries, no
raw-response or reasoning-trace persistence, no V134 or external-language access, and zero execution. The
automatic registered decision was:

> `neither_condition_realizes_controlled_boundary_close_current_local_branch`

That decision requires a qualified interpretation because the direct condition produced valid semantic
evidence while the thinking condition encountered a preregistered parser/template mismatch.

## Direct condition

Direct inference was close but did not qualify:

- structured validity: 100%;
- overall exact accuracy: 96%;
- clear-request accuracy: 100%;
- clarification-resolved accuracy: 100%;
- ambiguous-request abstention accuracy: 80%;
- full five-stage group accuracy: 80%;
- false-known rate over all non-known fixture truths: 3.33%.

All four errors occurred on intentionally ambiguous requests: two were mapped to `K01`, two to `U00`, and
sixteen correctly mapped to `A00`. The model therefore solved every request after decisive information was
present, including all valid-undeclared and unsupported cases, but it overcommitted on four of twenty cases
where the decisive distinction was absent.

That localized failure was enough to break the sequential safety gates. The deterministic policy queried on
80% rather than at least 95% of ambiguous groups. Its mean cost was 1.04, above the 0.70 ceiling; false-known
decisions on the non-known right truth reached 10%, and safe non-known decisions reached 90%. Asking still
improved cost by 0.685 versus the model's no-query output, but the missed queries were too consequential.

## Thinking condition: technical invalidity, not semantic failure

The thinking condition had zero valid final answers under the frozen parser, so invalid output mapped safely
to `A00`. Its resulting 20% apparent accuracy and perfect ambiguous abstention are not semantic performance
measurements.

Aggregate validation metadata—not raw responses—showed:

- 93 responses classified as `unclosed_thinking_trace`;
- seven classified as `invalid_final_json`;
- thinking-tag presence on 93%;
- seven responses at or near the 512-token ceiling;
- mean generated length of 229.22 tokens.

The pinned tokenizer template inserts the opening `<think>` tag into the prompt when thinking is enabled.
Generated suffixes therefore normally contain the closing `</think>` tag without repeating the opening tag.
V137's parser instead required equal opening and closing counts inside the generated suffix. The 93 closing-
tag responses were rejected by construction. Because raw traces were deliberately not persisted, they cannot
be reparsed after the fact.

Freeze the V137 thinking condition as technically invalid and do not use it to claim that thinking helps or
hurts semantic classification. Do not rerun the V137 test split. A recovery requires a separately locked
parser/template contract and a fresh population, such as the still-unused V135 development split.

## Efficiency and boundary

Direct inference used 683.60 generation seconds and 12.26 generated tokens per fixture on average. Thinking
used 2,182.51 seconds and 229.22 tokens per fixture—about 3.2 times the generation time and 18.7 times the
tokens. Peak active memory remained about 16.7 GB.

V137 is synthetic-development evidence only. The valid direct result shows that explicit capability
boundaries and targeted clarification can make the clear known/novel/unsupported distinctions easy for this
model, while ambiguous abstention remains the limiting mechanism. It does not establish external-language
transfer, human identifiability, unrestricted open-world understanding, independence, deployment safety,
induction, authority, action, or execution.
