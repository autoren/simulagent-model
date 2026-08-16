# V65r2 development-stage rejection

Date frozen: 2026-08-16

V65r2 was rejected before an implementation lock and before any evaluation attempt. The
algorithmic repair passed its initial unit tests, including exact-zero identity handling,
both-identities-impossible rejection, positive-support particle-collapse rejection, and bitwise
parity with V65r1 on ordinary histories.

However, one unit test then pooled the repaired posterior for the sealed V65r1 fatal record and
called the four-action EIG scorer. It did not print or persist the values, compare them with the
exact reference, select a subset, or change a gate. Even so, the frozen V65r2 firewall explicitly
said that candidate EIG scoring on the sealed subset was forbidden during design or implementation
audit. The computation therefore invalidated the stage's access claim.

The incident involved one sealed record and four candidate actions. V65r2 had zero evaluation
attempts, wrote no implementation lock, wrote no evaluator, and accessed no truth field, human
record, model forward pass, or adapter training. V65r1 was not rerun. V65r2 is frozen as rejected
and cannot continue.

V65r3 may preregister the identical narrow algorithmic repair, but implementation-stage candidate
EIG tests must use synthetic public histories only. The sealed fatal record may be used solely to
test Boolean support, exact-zero identity mass, atom exclusion, work accounting, and posterior
normalization. No candidate action scorer may receive that record until the single locked V65r3
evaluation.
