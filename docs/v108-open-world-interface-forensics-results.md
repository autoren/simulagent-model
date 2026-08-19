# V108 Typed Interface Forensics Result

## Outcome

The preregistered aggregate diagnostic passed every format-dominance and zero-access gate. It automatically
reparsed the exact 192 frozen V107 responses without reading development or protected-test language,
manually inspecting response text, loading a model, generating a token, or changing any V107 score.

The diagnosis is unambiguous:

- all 61 invalid observed responses used an exact, unique short intent name exposed by the visible catalog;
- no invalid observed response belonged to any other registered structural-failure category;
- all 61 became structurally valid after the single permitted local-name-to-qualified-ID mapping;
- the catalog contained zero ambiguous short intent names;
- counterfactual exact known-intent accuracy rose from `0.0%` to `79.69%`;
- counterfactual overall exact decision accuracy rose from `35.16%` to `75.0%`;
- status macro F1 rose from `46.91%` to `74.64%`;
- top-confidence 80% error fell from `56.31%` to `16.50%`.

This establishes that V107's zero accepted known-intent score was dominated by a typed serialization
mismatch. V105 displayed both a qualified `intent_id` and an unqualified `intent` for every declared
capability, while its validator accepted only the qualified form. The model consistently selected the
human-readable short field. The diagnostic did not infer an intent from language or gold labels, change
status or confidence, retry a response, or regenerate anything.

The counterfactual is not a replacement result and is not deployment evidence. In particular, mean
decision regret increased from `1.0977` to `1.3672` and false-known acceptance increased from `0%` to
`14.06%`. Nine novel-valid records had also been expressed as known short identifiers; accepting their
serialization exposes genuine semantic overcommitment that V107's fail-closed validator had previously
turned into abstention. Thus the diagnostic separates two problems: known grounding was much better than
V107's headline score suggested, but novelty-versus-known discrimination still needs work.

## Boundary and decision

V107 remains frozen, nonqualifying, and unchanged. Its protected test remains sealed. V108 emitted only
aggregate counts and metrics—no raw response or record identifier—and used zero model loads, generations,
API calls, training runs, service calls, tool executions, or external side effects.

The authorized successor is a fresh development-only constrained typed-interface study. It should expose
exactly one machine-accepted identifier representation, preferably by requiring selection from enumerated
typed IDs or by deterministically canonicalizing unique displayed aliases before validation. It must be
prospectively locked, rerun no V107 condition, preserve complete safe-hypothesis retention and fail-closed
abstention, and separately measure the newly visible false-known risk. Passing that study would justify a
richer sequential clarification benchmark; it would not by itself authorize the protected test, an API
model, training, or model control authority.
