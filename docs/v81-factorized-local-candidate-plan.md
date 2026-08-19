# V81 Factorized Local Candidate Plan

V80 established that the pinned local model can obey the broad safety boundary, but a direct
variable-length candidate list was not sufficiently selective or stable. V81 changes the interface,
not the frozen model: the model emits five fixed Boolean semantic-compatibility labels, and ordinary
deterministic code composes those labels into the four V79 interpretations plus the mandatory
`none_of_the_above` escape candidate.

This factorization keeps probabilities, belief updates, clarification selection, actions, tools, and
side effects outside the model. The model cannot invent candidate identifiers and cannot decide which
candidate should control execution. Out-of-ontology detection is explicit and must be logically
consistent with the two operation labels.

The evaluation population contains 24 new project-authored requests. No V80 request is reused. The
five strata and noncompensatory quality gates remain comparable, while new gates directly test exact
label vectors, mean label accuracy, and out-of-ontology classification. The prompt, population,
parser, deterministic composer, model revision, decoding settings, and gates must be sealed before
the first generation.

V81 permits one deterministic local generation per record, with no malformed-output retry. It
permits no API, adapter training, human-language collection, tool call, or side effect. Passing would
authorize only a new preregistration for mapping composed candidates into an explicitly calibrated
belief interface. Failing would stop this local candidate-integration line until a materially
different design is justified.
