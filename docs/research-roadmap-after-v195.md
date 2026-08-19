# Research roadmap after V195

## Evidence now established

V186–V190 established that the 14 frozen SGD capability contracts are finitely identifiable through typed questions
and that the fixed `domain -> normalized intent concept` hierarchy has a small protected identity-only cost advantage:
`0.38` versus `0.40` for generic clarification. That evidence contained no utterance-level language understanding.

V191–V195 added the missing development-language premise in four controlled layers:

1. **Fresh population:** V191 selected 84 dialogue-disjoint SGD development records, six per contract, plus 14 missing
   controls without reading language.
2. **Fail-closed interface:** V192/V193 exposed only observable language and a complete 14-option menu. Proposals can
   rank clarification options or abstain, but cannot accept, reject, register, prune, act, or execute.
3. **Strong model-free control:** V194's `CHAR_LAST` achieved primary top-3 recall `0.8541666667` and controller cost
   `0.2583333333`, making this a genuine incremental-model comparison rather than an LLM-versus-straw-man result.
4. **Incremental local-model value:** V195's frozen `Qwen3.8-27B-4bit` condition achieved primary top-3 recall
   `0.9291666667` and cost `0.2141666667`, improving on `CHAR_LAST` by `0.0441666667`. Structural validity, target
   retention, and trusted exactness were `1.0`; final truncation, retries, APIs, protected access, and authority were
   zero.

The reasoning-control finding is equally important: every low-effort reasoning phase exhausted its 48-token budget
and none naturally closed. Mechanical closure plus a separately reserved 64-token final phase yielded complete JSON
in every case. This exact runtime contract is part of the policy and must travel unchanged into confirmation.

## Supported claim boundary

V195 supports this development claim:

> On a finite 14-option menu and a fresh externally authored SGD development population, one fixed bounded local
> model can non-authoritatively rank trusted clarification questions more economically than both a fixed typed
> hierarchy and a strong character-retrieval control, while a complete authoritative fallback preserves the target.

It does not establish unrestricted open-world understanding, calibrated acceptance or rejection, ontology induction,
service-version identification, human-answer accuracy, deployment safety, action quality, or execution.

## Active Track A: unchanged fresh confirmation

The immediate successor must be confirmatory rather than another model or prompt comparison.

1. Audit remaining SGD candidate metadata without emitting utterances.
2. Freeze a fresh globally dialogue-disjoint population across all 14 contracts and missing controls. Exclude all
   dialogue IDs used by V183 and V191.
3. Extract only the selected observable conversation prefixes after the identities are locked.
4. Reuse unchanged: model repository and revision, quantization, final-user prompt, visible menu, temperature zero,
   `reasoning_effort=low`, 48-token reasoning cap, mechanical closure, reserved 64-token final cap, exact parser,
   no retries, top-3 controller, costs, and qualification gates.
5. Reconstruct the complete outcome from persisted normalized fixture artifacts and freeze a positive, negative, or
   non-incremental result. No prompt repair or alternate model follows a miss.

A confirmation pass supports only finite non-authoritative menu reduction. It does not authorize direct ontology or
action use.

## Subsequent tracks

### Track B — model-free robustness and shift

After confirmation, test which part of the gain survives controlled vocabulary, paraphrase, distractor, ambiguity,
and menu-expansion shifts. Freeze transformations before scoring. Compare the unchanged local policy to `CHAR_LAST`,
not just to the V190 hierarchy. Treat malformed and unsupported outputs as insufficient and retain every target.

### Track C — shadow ontology acquisition

Defer until Track B establishes stable finite-menu behavior. A model may propose provisional typed concepts only in a
shadow ledger. Deterministic type, precondition, effect, identifiability, and conflict checks must precede any sandbox
trial. Model confidence never registers or deletes a concept.

### Track D — clarification reliability and human factors

Deferred because independent human participants are unavailable. Simulated source answers or model-generated answers
may stress the planner but must remain labeled simulations. They cannot establish human reliability, response cost,
or cognitive burden.

### Track E — richer partially observable decisions

After the observation interface survives confirmation and robustness tests, couple semantic ambiguity to delayed
state-dependent consequences. Compare posterior-aware clarification and control against MAP, retrieval-only,
always-ask, and always-defer policies. The key question is then whether preserving semantic uncertainty changes action
value, not merely whether it improves a classification score.

### Track F — API or additional-model comparison

Not currently justified. The local condition already adds development value, and confirmation has higher information
value than another capacity point. An API or second model requires its own preregistered question, cost/accounting
boundary, fixed prompt, and no-fallback rule after Track A.

## Durable stop and safety rules

- Do not inspect confirmation language before population and policy locks.
- Do not weaken the V195 incremental gate, swap prompts, increase budgets, retry, or select among outputs after seeing
  confirmation performance.
- Do not treat top-3 inclusion, similarity, abstention, or model confidence as authority.
- Keep the complete 14-contract target set available behind trusted clarification.
- Do not use protected language, an API, training, ontology mutation, real service calls, side effects, action, or
  execution without a separate explicit lock and authorization.

