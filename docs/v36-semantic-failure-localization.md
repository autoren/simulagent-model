# V36 semantic failure localization

Status: descriptive analysis of the frozen V36 predictions. This analysis performed no fitting, model forwards, selection, threshold changes, or V32 evaluation scoring, and it does not amend the frozen decision.

## What transferred

Predicate, entity binding, exact atom, and relation order all scored 1.000. The language grounder therefore transferred the content of the embedded proposition perfectly across the V36 families.

## What failed

Lexical sign scored 0.768 and outer operation scored 0.768. Both failures are substantive: giving the compiler oracle sign while retaining the predicted operation raises truth accuracy only to 0.785, while giving it oracle operation with predicted sign raises it only to 0.807.

Of 271 operation errors, 200 (73.8%) defaulted to `unresolved`. This is a strong out-of-template fallback, not a single reversed label.

Operation accuracy was 0.931 without a distractor and 0.509 with one. Thus distractor sensitivity accounts for much of the operation failure, although the non-distractor score still misses the preregistered 0.950 gate.

All unresolved families reached 1.000 compiled truth even though their lexical-sign accuracy was imperfect. That is expected because unresolved propositions compile to unknown under either lexical sign; it is not evidence that sign transfer succeeded.

## Independence and interpretation

Exact evidence-text overlap between all V32 factor corpora and V36 was 0. The result is therefore consistent with template-local semantic readouts that fit V32/V35 development language but did not learn a stable representation of sign and discourse operation.

The justified next direction is restricted to the semantic interface: develop invariance across wording, distractor placement, negation scope, and operation paraphrase on new development-only language. Keep the backbone, atom/binding interface, executor, V32 evaluation firewall, and V28 prohibition fixed. Do not preregister or construct the end-to-end relational suite yet.

Machine-readable tables, including all confusion matrices and operation/sign cells, are in `outputs/v36-independent-confirmation/failure-localization.json`.
