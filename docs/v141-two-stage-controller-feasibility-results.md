# V141 Correlated Two-Stage Controller Feasibility Results

## Result

V141 establishes a conservative model-free reliability envelope for the two mechanisms required by V140.
It assumes no independence between finalizer validity, evidence-sufficiency judgment, and typed proposal
correctness. It also makes no independence assumption across the five records in a minimal-pair group.

At the registered reference point where all four marginal reliabilities are 99%, the Fréchet-bounded
controller guarantees:

- structural validity: 99%;
- clear and clarification-resolved accuracy: at least 97%;
- ambiguous abstention: at least 99%;
- overall accuracy: at least 97.4%;
- complete five-stage group accuracy: at least 87%;
- false-known rate on non-known fixture truths: at most 1%;
- sequential mean decision cost: at most 0.647;
- sequential improvement over no-query behavior: at least 0.393;
- difficult-branch false-known probability: at most 4%;
- difficult-branch safe non-known probability: at least 96%.

Every inherited gate that can be bounded from the registered marginals passes. Incorrect typed proposals
are charged the worst frozen decision cost, and query/final-decision failures may be adversarially aligned.

The minimum symmetric marginal reliability on the exact 0.001 grid is 99%. Holding the other marginals at
99%, the one-at-a-time thresholds are:

- bounded-finalizer validity: 99%;
- ambiguity sensitivity: 98%;
- decidable-case specificity: 98.5%;
- typed-proposal correctness: 98.5%.

The sequential cost gate, rather than the basic 95% ambiguity gate, raises the required ambiguity
sensitivity to 98%: a missed query can expose the system to a costly wrong-known decision.

## Limitation and decision

Candidate attraction among the residual errors cannot be derived from marginal reliabilities and is not
certified. It remains a mandatory empirical gate. Nor does V141 show that any current LLM realizes these
reliabilities; V139 did not.

Freeze V141 as positive abstract feasibility. The next authorized step is a fresh interface and population
design that makes the bounded finalizer and evidence-sufficiency certificate independently measurable. It
does not authorize a language/model run. Same-model stages must remain modeled as correlated, raw reasoning
must not be persisted, and invalid output must fail closed. V134, external language, APIs, training,
induction, authority, action, and execution remain closed.
