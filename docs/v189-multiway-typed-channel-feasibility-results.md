# V189: Multiway typed-channel feasibility result

## Bottom line

V189 is formally failed because one preregistered control gate was specified in the wrong direction. The computation itself is valid and produced an important development pattern: even under conservative zero-overhead bit-slot pricing, a fixed two-turn `domain → intent` menu costs 0.38, strictly below the 0.40 generic clarification.

That pattern is development evidence only. V189 cannot be relabeled positive, and its gate cannot be edited after seeing the result.

## What the policies did

All 11 pricing scenarios were complete, exact, and target-retaining. No language, protected data, model, API, training, authority, or execution was used.

Under pure worst-case bit-slot pricing (`o=0`):

- exact and best open loop both used `domain → intent concept`;
- mean cost was `0.38`;
- mean turns were `1.8` because singleton domains stopped after one answer;
- typed-only completion was `1.0`;
- final exactness and target retention were `1.0`;
- adaptive advantage over open loop was `0.0`.

The coarse-only domain/transaction policy chose generic clarification at cost 0.40. Therefore the strict 0.02 gain required the within-domain intent menu; domain alone was not enough at zero turn overhead.

As turn overhead increased, multiway compression became progressively cheaper. At overhead 0.01 the full policy cost 0.36. From overhead 0.03 onward, a single global intent menu was cheaper than the two-turn hierarchy; its cost fell from 0.31 to 0.13 across the grid. The entropy lower-bound condition cost `0.363425`, equal to 0.10 times the prior entropy.

No scenario exhibited adaptive advantage: exact and best fixed open loop always tied. This is menu compression, not evidence for history-dependent planning.

## Why the formal gate failed

The protocol simultaneously defined robust multiway value as strict improvement under pure bit-slot pricing and required the pure bit-slot policy to be **not below** generic clarification. Those conditions are incompatible whenever robust value exists.

The observed pure-bit cost was `0.3800000000`, so the `requiredPureBitSlotAllQuestionCostNotBelowGeneric` gate failed. The other gates passed. The runner therefore followed the frozen failure branch even though its summary correctly recorded `robust_multiway_value: true`.

This is a protocol-design error, not an implementation or arithmetic error. It cannot be repaired by flipping the gate inside V189, because the direction would be changed after observing the outcome.

## Scientific interpretation

The development pattern supports a narrower hypothesis than originally intended:

> A hierarchical categorical menu can encode the 14 finite contracts in fewer charged bit slots on average than a flat four-bit generic answer, because the development prior and domain hierarchy permit early stopping.

This does not establish open-world language understanding, human usability, answer reliability, or adaptive decision value. The global intent menu also assumes the correct concept is present and comprehensible.

## Decision

Freeze V189 as a failed formal experiment with valid exploratory development evidence. Do not edit its gate, rerun it, or claim robust feasibility from it.

The only rigorous successor is a separately preregistered fresh confirmation using the already-sealed protected identity role without reading its utterance language. It must freeze the V189 questions, pure bit-slot pricing, `domain → intent` policy, generic comparator, and a coherent strict-improvement gate before scoring the protected target distribution. That confirmation would test menu compression only; it must not claim adaptive planning.
