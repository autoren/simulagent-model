# V129 Complete Clarification-interface Audit Results

## Outcome

V129 failed the preregistered all-prior interface gates:

> `complete_typed_clarification_infeasible_keep_language_and_model_closed`

The model-free census evaluated every one of the 66 safe truth/presented-known-candidate pairs. A perfect
typed answer produced 100% known and unsupported decisions under every prior and error regime, confirming
that the full-catalog interface is identifying and implemented correctly.

At 95% reliability, the full interface was strongly better at recovering alternative known intents under
uniform and moderate priors. Channel-aware exact-known probability was 95% for symmetric and
abstention-attracted errors, and 79.17%--95% under candidate attraction. Regret was 0.8350--1.1242 in the
uniform and moderate conditions.

The strong 75% candidate prior remained decisive. Under symmetric and candidate-attracted errors,
exact-known probability was only 15.83%; regret was 1.2323 and 1.4847. The correct alternative answer
raised its posterior substantially but not enough to cross the roughly 90% posterior threshold imposed by
the frozen asymmetric loss matrix. Candidate attraction also made the complete interface worse in regret
than the frozen candidate-specific comparator. A planner that incorrectly assumed symmetric error failed
the strong-prior regret gate as well.

False-known probability remained below 4.45% and unsupported correctness was 95% at the required
reliability, but the known and regret failures are noncompensatory. Freeze V129 negative. Do not interpret
the result as evidence that a real human or model can answer the full menu at 95%, and do not run such a
channel yet.

The result localizes the next question: the interface contains the necessary answer, but one noisy answer
does not provide enough Bayes factor to overcome a strongly concentrated wrong prior under the frozen
losses. A successor may only derive the minimum reliability or number of genuinely independent pieces of
evidence needed across the three error regimes. It must remain model-free and cannot assume that repeated
LLM samples are independent. Language/model access, protected data, capability induction, richer planning,
APIs, training, authority, and execution remain closed.
