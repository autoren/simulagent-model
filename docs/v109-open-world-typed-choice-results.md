# V109 Single-Choice Typed Interface Result

## Outcome

The single-choice interface passed every mechanical and access gate but failed the semantic gate set. The
pinned local Qwen3.8-27B 4-bit model completed one 192-generation, zero-temperature, no-retry condition on
128 model-blind development holdback records and 64 missing-observation controls. Runtime was 1,814.6
seconds (30.2 minutes), with one model load and 17.01 GB peak active memory.

The interface result is conclusive:

- structured validity was 100% overall and on observed language (`192/192` and `128/128`);
- missing-observation abstention was 100% (`64/64`);
- exact known-intent accuracy was 96.88%;
- familiar-known accuracy was 100%, and unfamiliar-known accuracy was 93.75%;
- exact overall decision accuracy was 70.31%;
- unsupported accuracy, recall, and precision were each 87.5%.

Thus the typed choice codes fully removed the serialization failure diagnosed by V108. The model can map
both familiar and lexically unfamiliar supported requests to the exact known capability when the machine
interface makes the permitted output unambiguous.

The residual open-world result is equally sharp. The model selected no `NOVEL` choice for any record:

- novel-valid exact accuracy and recall were 0%;
- the 32 novel-valid requests became 13 `KNOWN`, 3 `UNSUPPORTED`, and 16 `ABSTAIN` predictions;
- false-known acceptance on non-known requests was 20.31%, above the 10% gate;
- status macro F1 was 58.90%;
- 10-bin confidence ECE was 0.2538;
- top-confidence 80% error was 25.24%;
- mean decision regret was `1.3828`, or `0.2578` above ask-always, narrowly failing the frozen `0.25`
  relative-regret limit.

The model therefore distinguishes ordinary known capabilities and wholly unsupported requests much better
than it distinguishes a valid but unlisted capability from the nearest listed one. It is also too confident
when making that distinction. This is no longer plausibly a JSON or identifier problem.

## Boundary and decision

The protected test remained sealed. V109 used no API, training, manual utterance inspection, service call,
tool execution, or external side effect. Every choice remained a non-authoritative shadow proposal; all 17
safe hypotheses were retained, and deterministic code alone expanded a valid code into a typed decision.

Freeze V109 as a positive interface result and a negative novelty/uncertainty result. Do not retry it,
change the prompt or thresholds after seeing the outcome, run the protected test, add another model, or
train an adapter under this branch.

The next benchmark should make clarification sequential and decision-relevant. A model proposal can inform
but never determine the belief state. The planner should choose among asking a targeted distinction,
requesting missing details, routing a candidate to sandbox review, rejecting, or taking a reversible known
action, with explicit costs and delayed consequences. The benchmark must test whether preserving the
known-versus-novel alternatives and purchasing targeted evidence reduces the costly false-known decisions
that V109 exposed.
