# V124 Schema-Guided Dialogue Source-feasibility Plan

V124 evaluates the official archived Schema-Guided Dialogue repository as a new controlled open-set source.
The revision and archive URL are pinned before payload download. The repository documents over 20,000
annotated conversations, service/intent schemas, evaluation services and domains not seen in training, and
a CC BY-SA 4.0 license.

After the design is frozen, one automatic download and parse may produce a text-free structural inventory.
Candidate units are user turns with exactly one explicit `INFORM_INTENT` action matching one active intent.
Test candidates are categorized relative to train schemas as exact known service/intent, valid unseen service
within a seen domain, or unseen-domain unsupported. This categorization is an evaluation construction, not
runtime evidence.

No utterance, slot value, individual dialogue, or raw response may be emitted or manually inspected. No
retrieval system, trigger, model, policy, protected set, induction, authority, or execution is evaluated.
A pass authorizes only a separately locked text-free catalog and population design.
