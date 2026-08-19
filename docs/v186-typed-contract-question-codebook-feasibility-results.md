# V186: Typed contract question-codebook feasibility

## Bottom line

V186 is a clean positive feasibility result. A finite, text-free clarification codebook derived only from the 14 frozen V183 semantic capability contracts distinguished every contract from every other contract. This authorizes a separately locked clean exact-planning comparison, but it does not yet show that the questions are cheap, natural, robust to answer error, or useful for raw-language understanding.

## Result

- Contract universe: 14 (`6 KNOWN`, `6 PROVISIONAL`, `2 UNSUPPORTED`).
- Exhaustive contract-pair census: 91 of 91 pairs separable.
- Minimum number of separating questions for any pair: 5.
- Distinct complete answer vectors: 14.
- Equivalence classes: 14 singleton classes; largest class size 1.
- Question vocabulary: 164 non-invariant binary questions across six families.
  - normalized intent concept: 14
  - declared domain: 8
  - any service slot: 59
  - required slot: 23
  - result slot: 59
  - transactionality: 1
- Development binding: 120 observed records reconstructed exactly; 12 missing observations remained `INSUFFICIENT`.
- Protected binding: 120 observed records reconstructed exactly; 12 missing observations remained `INSUFFICIENT`.
- Development/protected record-ID overlap: 0.

All feasibility and safety gates passed. No utterance or dialogue language was read, no planner was scored, no model or API was used, and no registration, trusted-state mutation, service call, side effect, action, or execution occurred.

## Interpretation

The V185 failure was not caused by an intrinsically non-identifying ontology. The frozen contracts contain enough prospectively askable typed structure to distinguish the hypotheses exactly. The failure was specific to passive schema-similarity signals: those signals could not choose a high-precision shortcut, whereas direct clarification about semantic attributes can in principle identify the target.

This is an existence result, not a deployable language result. V186 used dataset-provided source annotations as oracle answers in a simulation. Those answers are not human evidence, model evidence, or deployed-sensor evidence. The full 164-question inventory is deliberately exhaustive and redundant; a planner must select a small adaptive subset rather than administer it wholesale.

Normalized intent-concept questions are especially direct and may make this finite universe easy. That is legitimate for the present control experiment, but it also limits the claim: V186 does not establish open-world discovery outside the 14-contract universe, user comprehension of question wording, or robust semantic answer extraction.

## Decision

Freeze V186 as a positive codebook-feasibility result. Proceed only through a separately preregistered clean exact-planning comparison with explicit costs and the frozen codebook. Compare exact adaptive selection against open-loop, greedy information gain, source-order, always-generic clarification, immediate deferral, and a target-informed oracle. Keep protected utterance language sealed.

After the clean comparison, stress common-shock and adversarially correlated errors before considering any learned answer proposer. A model remains optional and non-authoritative and requires separate authorization.
