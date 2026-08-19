# V197 protected confirmation language projection plan

## Purpose

V184 stored its protected role as one 132-record JSON artifact. V196 selected 125 records after removing dialogue
overlap and duplicates. V197 creates an exact sanitized projection before any scoring or model use so the eventual
confirmation runner need not parse or retain the seven excluded conversations.

## Exact projection

Read the monolithic protected artifact once after this design is locked. Select records solely by the frozen V196
opaque ID set. Emit only record ID, confirmation role, observation availability, and exact conversation. Missing
controls must retain null conversations. No language may affect selection.

The projection must contain 113 observed records and 12 missing controls. Seven source records are necessarily read by
the deterministic projector but are neither emitted nor scored. The output excludes presented candidates and every
gold or evaluation field.

## Boundary

V197 runs no retrieval, policy, model, API, training, ontology change, trusted mutation, service, action, or execution.
A pass authorizes only a separate preregistration of the unchanged V195 policy on the 125-record projection.

