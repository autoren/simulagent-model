# V154 adaptive local question-order plan

## Scientific question

V154 asks whether the pinned local Qwen3.8-27B model can reduce trusted clarification cost when its only output is an ordering of six registered questions. It cannot propose states, express confidence, prune the complete hypothesis universe, answer a question, authorize catalog membership, act, or execute.

The 96 frozen V152 development request fixtures produce 120 sequential episodes. Irrelevant questions return no trusted selection and remain `A00`; the discriminating question uses the immutable V152 answer event and typed-witness firewall. Consequently, a bad ranking can increase question cost but cannot change final acceptance.

## Prospectively adaptive model conditions

The primary condition is direct decoding with thinking disabled and at most 64 new tokens. If and only if the direct condition fails any qualification gate, the same already loaded model runs the registered bounded-low-reasoning challenger.

The challenger uses the pinned official chat template with `reasoning_effort="low"`. It is not allowed to consume the whole response budget in a reasoning loop. The runner gives the reasoning phase at most 48 generated tokens, retains tokens only through the first natural `</think>` if present, otherwise injects the close tag, and then starts a separate 64-token final continuation. Only the final continuation is parsed. This reserves a final-answer budget mechanically rather than relying solely on a prompt request. Both phases are one preregistered inference path, not independent votes or retries.

Direct success stops the study before the challenger, avoiding an unnecessary second condition. If direct fails, the challenger is evaluated under the same gates. No result-dependent prompt, budget, threshold, parser, or fallback change is permitted.

## Output and fallback

The only valid response is a strict JSON object whose sole key is `query_ranking`, containing all six registered query IDs exactly once. Invalid or incomplete output receives the frozen source-order fallback. Raw prompts, reasoning traces, and final responses are hashed but never persisted or manually inspected.

## Gates

Each observed condition independently requires at least 98% structural validity, 90% top-1 query accuracy, 0.95 MRR, mean correct rank at most 1.25, mean sequential cost at most 0.45, and improvement over no-query of at least 0.55. Final exact accuracy after trusted answers, irrelevant-intermediate fail-closed rate, and authoritative hypothesis retention must all be 100%; candidate-proposal fields and execution must remain zero.

The access envelope permits one model/tokenizer load, 96 direct calls, and—only after direct failure—96 bounded reasoning plus 96 bounded final calls. It prohibits retries, closed-answer or evaluation generation, raw-output inspection or persistence, APIs, training, services, side effects, and execution.

Passing authorizes only a separate preregistration for the selected condition on the sealed V152 evaluation split. It does not authorize immediate evaluation, calibration, tuning, induction, authority, action, or execution. Failure closes both question-order conditions on this population.
