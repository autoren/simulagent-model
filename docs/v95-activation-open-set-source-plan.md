# V95 Activation-Turn Open-Set Source Plan

## Material change from V94

V94 established that a global multi-service capability catalog is structurally viable, but accumulated
dialogue history made unfamiliar known requests too rare and the pair partition could hide only one
intent while retaining four declared intents. V95 does not reuse V94 language or relax its outcome. It
uses the untouched `dev/dialogues_006.json` shard and changes the construction in two substantive ways:

1. only source-annotated **intent activation turns** can supply known, novel-valid, or unsupported
   examples; and
2. familiar versus unfamiliar is defined using only the current user turn, not accumulated history.

An activation is a non-`NONE` source state whose active intent differs from the previous valid state for
that service in the same dialogue. Continuations that merely preserve an already-active intent cannot
enter those three classes. Genuine source `NONE` states remain the only insufficient-evidence source.

## Service-stratified hidden pairs

Before any language-derived feature is used, source-supported fresh services are hash-ordered. One is
withheld completely for unsupported requests. Within the remaining catalog, two different services are
hash-selected; exactly one source-supported intent pair from each is hidden. All other source-supported
pairs are declared known, and at least three declared pairs must remain.

The five text-free classes are therefore:

- familiar declared intent activation;
- zero-overlap unfamiliar declared intent activation;
- source-valid activation of one of two service-stratified hidden intents;
- intent activation from a hash-withheld unsupported service;
- source `NONE` state as insufficient evidence.

## Source-stage boundary

The one-shot inventory may automatically tokenize the current turn to count schema overlap, but it may
not emit language, tokens, slot values, histories, or prompts. Passing authorizes only a later,
dialogue-disjoint population and calibration/evaluation preregistration. It does not authorize language
extraction, manual inspection, local or API model access, training, posterior integration, planning, or
execution.
