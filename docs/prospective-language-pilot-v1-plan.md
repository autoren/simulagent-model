# Prospective language pilot V1 — protocol and testing plan

## Decision and claim boundary

A qualified external speaker channel became available after the V224 stopping rule: the user volunteered to author
prospective requests and answer later clarification questions. This authorizes construction and Phase 1 use of a
single-participant controlled pilot. It does not retroactively alter V224 and does not yet authorize a model run.

Phase 1 can support only this claim:

> One external participant prospectively authored and immutably locked initial natural-language requests for 16
> project-authored controlled scenarios before seeing any scored assistant output.

Phase 1 cannot support assistant understanding, open-world task success, population generalization, independent
ontology authorship, or clarification quality. Those require later frozen phases and remain unresolved.

## Research question

Can the existing uncertainty-aware architecture be evaluated as a language-guided decision system across familiar,
preference-sensitive, sensitive, creative, evidential, and novel-rule domains without forcing every request into a
single static intent label?

The immediate objective is narrower: establish a usable, contamination-resistant observation channel for initial
human requests.

## Scenario population

The frozen population contains 16 controlled scenarios. Record IDs are opaque and participant order is a
deterministic hash permutation of participant code and study salt.

| Coverage family | Included settings |
|---|---|
| Everyday and logistical | groceries, clothing, community event, accessible transportation |
| Emotional and relational | neighbour request, friendship repair, grief support |
| Faith and Christian/theological | church meal rota, mixed-background Bible discussion, baptism explanation |
| Art and design | animation storyboard, renter-friendly reading corner |
| Fantasy, mystery, and pretend | wizard tower, clockwork mystery, lunar resource allocation |
| Information application | community photo archive and exhibition |

The participant sees a setting, private goal, and known facts. The later assistant receives only the locked request
plus a limited assistant-visible domain/action context. Research metadata, private goals, and known facts are excluded
from the public projection.

## Phase sequence

### Phase 0 — frozen infrastructure

- Freeze `configs/prospective-language-pilot-v1.json`.
- Validate 16 unique opaque IDs and required scenario fields.
- Hash the full config when the participant session begins.
- Refuse to resume a session if the config hash changes.
- Use pseudonymous participant codes; collect no name, email, credentials, or private real-world data.

### Phase 1 — all initial requests

This is the only currently authorized data-collection phase.

1. Show one scenario card at a time in randomized frozen order.
2. Collect the participant's natural first message or an explicit unable-to-respond disposition.
3. Require an own-wording/immutability attestation.
4. Atomically lock the record and immediately regenerate separate public, private, and audit projections.
5. Show no assistant output for any record until all 16 records are locked.
6. Stop at `phase_1_complete_waiting_for_assistant_run`.

No local model, API, assistant response, action, or sandbox execution occurs in Phase 1.

### Phase 2 — completed negative development run

After the passing Phase 1 audit, the exact local interactive condition was frozen and run once. The bounded local
proposal model plus deterministic controller chose exactly one of:

- one clarification message containing at most two closely related questions;
- an immediate sandbox-only plan; or
- defer.

The condition failed its structural-validity and truncation gates and remains a negative result. It was not retried.
See the Phase 2 plan and result documents. LLM-only and deterministic baselines remain deferred and must not ask the
participant competing questions for the same record.

### Phase 3 — active exploratory valid-only clarification batch

The failed Phase 2 condition produced 11 structurally valid non-fallback clarification records. A separate protocol
review hash-locked those exact questions for exploratory presentation without reclassifying Phase 2 as passing.
For each record, show the original scenario, private goal, locked request, and actual frozen assistant questions.
Collect and lock one natural participant response. Invalid, plan, defer, candidate-goal, and reasoning outputs remain
hidden.

### Phase 4 — terminal batch run (not yet authorized)

After clarification, the assistant must return either a sandbox plan or defer. It may not ask another question in V1.
No real purchase, message, booking, account mutation, or physical action is allowed.

### Phase 5 — participant evaluation (not yet authorized)

For every final record collect:

- understood goal: yes / partly / no / cannot tell;
- plan would accomplish goal: yes / no / cannot tell;
- unacceptable constraint violation: yes / no / cannot tell;
- clarification useful: yes / partly / no / not asked;
- optional short explanation.

## File and access separation

Runtime records live under the ignored directory:

```text
data/prospective-language-pilot/<study-id>/<participant-code>/
├── public/
│   └── phase1_initial_requests.jsonl
├── private/
│   ├── session_state.json
│   └── phase1_private_records.jsonl
└── audit/
    └── phase1_manifest.json
```

Only the public JSONL may be supplied to a later evaluated assistant. It contains the pseudonymous participant code,
opaque record ID, limited assistant-visible context, locked initial request, timestamp, and payload hash. The private
projection contains the participant card and research metadata. The audit manifest hashes both projections and
records an assistant-generation count of zero.

## Phase 1 acceptance gates

All must pass before any assistant batch is considered:

1. 16/16 records are either locked requests or explicit unable dispositions.
2. Config hash matches the session-start hash.
3. Public projection contains no participant cards, private goals, or research metadata.
4. Every stored public record recomputes to its locked payload hash.
5. Public, private, and audit file hashes match the final manifest.
6. Assistant generation count is exactly zero.
7. No assistant output was shown during initial collection.
8. Participant reports that the instructions and scenarios were understandable enough to complete the batch.

Failure of any integrity gate freezes the run for repair. An unclear scenario remains an explicit outcome and should
not be silently rewritten after language collection begins.

## Later scientific endpoints

If later phases are authorized, primary endpoints should be consequential rather than top-1 intent accuracy:

- participant-confirmed goal understanding;
- hard-constraint violation rate;
- successful sandbox-plan rate;
- harmful false-assumption rate;
- necessary versus unnecessary clarification;
- clarification utility;
- defer appropriateness;
- candidate-set recall at fixed budget;
- downstream decision regret where an executable scorer exists.

Creative and theological records admit plural reasonable outputs. They require constraint satisfaction, balanced
representation, and participant acceptability rather than one exact reference answer.

## Limitations

- The first pilot has one speaker.
- Scenario semantics are project-authored even though participant language is external.
- The participant is familiar with the broad research motivation.
- Several scenarios have subjective or plural valid outcomes.
- A later model run will be development evidence until independently replicated.

These limitations do not invalidate the usability and mechanism pilot. They prevent population-level or unrestricted
language-understanding claims.

## Verification commands

```bash
PYTHONPATH=python .venv/bin/python -m unittest python/test_prospective_language_pilot.py
.venv/bin/python -m pip install -r requirements-pilot-ui.txt
./scripts/run-prospective-language-pilot.sh
PYTHONPATH=python .venv/bin/python python/verify_prospective_language_pilot_phase1.py \
  data/prospective-language-pilot/prospective-language-pilot-v1/P001
```

The UI should be manually checked in a browser for landing, start/resume, scenario rendering, validation, immutable
locking, progress, completion, and the three separate downloads.
