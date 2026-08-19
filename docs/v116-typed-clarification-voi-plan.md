# V116 Typed Clarification Value-of-Information Audit Plan

V115 showed that a second static review of the same request and catalog is not new information. V116 asks
the narrower prerequisite question: if an external responder supplies a typed clarification answer, is that
answer valuable enough to justify a later controlled multi-turn benchmark?

The audit uses only V115's frozen structural population, candidate identifiers, the 17-choice catalog, and
the frozen decision costs. It reads no fresh or protected language, runs no model, and stores no individual
diagnostics. A response can confirm the candidate, select another declared intent, identify a valid
undeclared capability in one visible scenario, report that the request is outside all visible scenarios, or
abstain. The LLM neither asks nor interprets the answer.

Because no real responder is available, the answer channel is explicitly simulated rather than presented as
human evidence. Correct-answer probability is swept from 0.70 through 1.00. Half of non-correct mass maps
to insufficient evidence and the remainder is spread over wrong actionable answers. Three priors—uniform,
moderately candidate-biased, and strongly candidate-biased—stress posterior sensitivity.

The frozen Bayes policy retains all 17 hypotheses and chooses only a known shadow action, unsupported shadow
action, or abstention. Novel and insufficient answers always abstain. The primary feasibility gate requires
two conditionally independent typed answers at 95% reliability to beat the exact historical V115/V112-policy
mean regret under every prior while maintaining known coverage, unsupported coverage, and false-known
safety. A fully correlated two-answer condition is mandatory as a stress test.

A pass is only design feasibility under stated simulator assumptions. It cannot authorize new language or
model generation, protected access, induction, richer planning, API use, training, capability registration,
action authority, or execution.
