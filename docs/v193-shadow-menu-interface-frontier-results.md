# V193 shadow menu interface and oracle frontier results

## Verdict

V193 passed every finite-menu, prior-reproduction, parser, trusted-controller, oracle, target-retention, economics, and
access gate. It establishes a coherent non-authoritative language-component interface and shows that useful proposal
quality is achievable in principle. It does not score language or evaluate a model.

## Interface and safety

The visible menu contains 14 opaque options, eight distinct domains, and 14 distinct normalized intent concepts. The
option-to-contract mapping is held separately. Contract IDs and historical truth kinds are not proposer inputs.

All 12 malformed controls—including truncated JSON, unknown or duplicate IDs, wrong lengths, extra confidence fields,
wrong status, and wrong container type—mapped to `INSUFFICIENT`. Missing observations also map to `INSUFFICIENT`.
Invalid or insufficient proposals use the unchanged V190 hierarchy. A valid ranked proposal can only choose which
trusted menu question to ask; it cannot determine the terminal state or prune the candidate universe.

## Economics

Using the frozen V186 development frequencies reproduced V190 exactly:

- fixed domain-to-intent hierarchy: `0.3800000000` mean cost;
- always generic: `0.40`;
- target-informed top-1 oracle: `0.10`; and
- target-informed top-3 oracle: `0.20`.

The trusted top-1 question contains one proposal plus `OTHER` and costs `0.10`. Top-3 contains three proposals plus
`OTHER` and costs `0.20`. A miss receives trusted `OTHER` and then generic clarification at an additional `0.40`.

On the fixed 0.001 recall grid:

- top-1 first beats `0.38` at recall `0.301` and reaches the required `0.02` improvement at `0.35`;
- top-3 first beats `0.38` at recall `0.551` and reaches the required `0.02` improvement at `0.60`.

Target retention and oracle exactness are `1.0` because every question includes `OTHER` and exact terminal state comes
only from a trusted answer.

## Claim boundary and next step

This is a finite benchmark-menu result. It is not open-set or unrestricted open-world recognition, utterance
understanding, model performance, protected confirmation, or human-reliability evidence.

Freeze:

`freeze_V193_and_authorize_one_separately_preregistered_deterministic_language_ranker_evaluation_only`

The next study may preregister fixed deterministic rankings of V192 development language against the visible menu and
score top-1/top-3 recall, mean clarification cost, false narrowing, class-conditioned performance, and target
retention. It may not fit thresholds after scoring, run a local/API model, open protected language, register or prune
concepts, mutate trusted state, act, or execute.
