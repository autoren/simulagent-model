# V114 Record-Disjoint Rescued-Policy Transfer Plan

V114 tests the exact V113 rescue rule on 192 new MASSIVE records selected before language extraction. It
uses the unused remainder of the official test partition but excludes every V101 identifier—including all
256 sealed protected records—and every V112 identifier. The result is record-disjoint within the same
source distribution; it is not a cross-dataset or unrestricted open-world test.

The model condition, typed-choice interface, prompt, decoding, base policy, calibrated novelty evidence,
comparators, metrics, seventeen quality gates, and 48 missing-observation controls are inherited exactly
from V112. The only registered addition is V113's selected known-disagreement rescue: accept the LLM's
known intent when its retrieval score is at least 0.60 and trails the nearest intent by no more than 0.15,
using frozen confidence 0.75.

The same model response and retrieval observation feed both paired policies. V114 makes two separate
decisions: whether the complete rescued policy passes all seventeen absolute gates, and whether the V113
mechanism itself encountered enough opportunity and helped. A mechanism conclusion requires at least eight
eligible disagreements and four triggered rescues. It must correct at least 75% of changed records, yield
at least one net correction, not worsen regret, known accuracy, selective error, false-known acceptance, or
unsupported precision/recall, and leave novelty evidence exactly unchanged. Fewer opportunities make the
mechanism result inconclusive even if the full policy passes.

The frozen report includes the full paired correctness transition table, eligible and triggered counts,
corrected and introduced errors, rescue precision, net corrections, paired metric deltas, familiar versus
unfamiliar effects, and aggregate intent/scenario concentration. No threshold may be adjusted after the
population is selected.

Novel evidence remains `ABSTAIN` and still asks for clarification. All safe hypotheses remain retained.
Even a complete pass permits only a separate sandboxed typed-induction feasibility design; it does not open
the original protected set or authorize induction, planning, APIs, training, actions, tools, or execution.
