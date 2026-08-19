# V124 Schema-Guided Dialogue Source-feasibility Results

## Outcome

V124 passed:

> `freeze_positive_SGD_controlled_open_set_source_feasibility`

The pinned official archive contains 22,825 dialogues across 20 domains, 45 services, and 88 partitioned
service-intent definitions. The automatic parser found 56,831 user turns with exactly one `INFORM_INTENT`
action matching one active non-`NONE` intent: 39,760 train, 5,966 development, and 11,105 test candidates.

Relative to train schemas, the untouched test candidates divide into:

| Controlled class | Candidates | Domain coverage |
| --- | ---: | ---: |
| exact train service/intent (`known`) | 3,101 | 6 |
| unseen service in a train-seen domain (`novel_valid`) | 6,601 | 11 |
| domain absent from train (`unsupported`) | 1,403 | 4 |

All source gates passed. The archive revision and CC BY-SA 4.0 license are pinned. Candidate identifiers are
unique and schema-valid. The emitted inventory contains structural service, intent, domain, partition, and
class fields only—no utterances, dialogues, tokens, slot values, or raw language.

The run used one payload download and one automatic parse, with no manual language inspection, model load,
generation, API call, training, real service call, authority, or execution.

This establishes source feasibility, not retrieval or policy performance. The only permitted successor is
a separately locked text-free catalog/population design that selects train declarations and fresh evaluation
identifiers before any selected utterance is extracted. Signal evaluation, model use, induction, protected
access, and execution remain closed.
