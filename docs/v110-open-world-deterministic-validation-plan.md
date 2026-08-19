# V110 Deterministic Novelty, Abstention, and LLM Validation Plan

V110 addresses the prerequisite that V109 did not meet: reliable novelty detection and abstention. It is
a secondary development analysis over exact frozen model outputs, not a retry, new prompt, new model, or
protected-test evaluation. Before reading record-level inputs, the 128 V109 observed holdback records are
hash-split into 64 calibration and 64 evaluation records, exactly 16 per class in each subset.

The study compares the required controls: complete safe enumeration, ask-always, identifier grammar,
character n-gram retrieval, direct LLM classification, calibrated confidence abstention, a deterministic
retrieval novelty override, conservative LLM-plus-validation, and an oracle. Only abstention and retrieval
thresholds may be selected on calibration. The primary LLM-plus-validation rule is fixed: retrieval may
route a request to a visible-scenario novelty hypothesis; otherwise a decision is accepted only when the
LLM and retrieval agree exactly, with disagreement mapped to zero-confidence abstention.

The evaluation gates are noncompensatory. Aggregate accuracy cannot compensate for novelty failure,
false-known acceptance, overconfidence, unsafe shadow proposals, or decision regret. Every policy remains
shadow-only, the complete 17-hypothesis universe is retained, and actual execution count is deterministically
zero. A pass would authorize only a separately locked protected-test protocol for this exact deterministic
layer. It would not authorize protected access immediately, schema induction, an API, training, planning
authority, tools, services, or side effects. A failure keeps schema induction and richer sequential planning
closed under the long-running objective.
