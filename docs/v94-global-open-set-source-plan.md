# V94 Global Capability-Catalog Source Plan

## Material change from V93

V93 required three intents inside each service, but the pinned SGD development schema contains no
service with more than two. V94 does not relax or reuse V93. It uses the untouched
`dev/dialogues_005.json` shard and makes the agent-wide capability catalog, rather than an individual
service, the unit of open-set partitioning.

Before language-derived features are used, source-supported fresh services are ordered by SHA-256. A
fixed fraction is withheld completely to provide unsupported requests. Within the remaining catalog
services, source-supported service/intent pairs are separately hash-ordered and a fixed fraction is
hidden to provide source-valid novel intents. All other catalog intents are declared known.

The five text-free classes remain:

- familiar declared intent;
- zero-overlap unfamiliar declared intent;
- source-valid hash-hidden intent;
- request from a hash-withheld unsupported service;
- source `NONE` state as insufficient evidence.

## Source-stage boundary

The inventory can automatically tokenize language to compute overlap counts, but it cannot emit
language, derived tokens, values, histories, or prompts. Passing authorizes only a later, dialogue-
disjoint population and calibration/evaluation preregistration. It does not authorize language
extraction, model access, API use, training, posterior integration, or execution.
