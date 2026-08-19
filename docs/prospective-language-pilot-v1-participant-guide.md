# Participant guide — prospective language pilot V1

## What you will do

You will see 16 scenario cards covering everyday decisions, relationships, faith, art, logistics, mysteries, and
fictional situations. For each card, write the first message you would naturally send to an assistant.

Complete all 16 initial requests before any assistant responses are generated. Expect approximately 30–45 minutes,
but you may close the browser and resume later using the same participant code.

## Writing rule

Read the full card, then use your own wording. Do not copy the private-goal sentence, label the intent, enumerate every
possible interpretation, or deliberately try to help or trick the assistant. You may omit facts you would not
naturally mention in a first message.

Do not enter your name, email, credentials, exact address, medical records, or other real private information. Every
scenario is hypothetical and no real-world action will occur.

If you genuinely cannot form a natural request, select the unable-to-respond option and give the closest reason. That
is a valid research outcome.

## What locking means

Each submission is immutable. The app creates:

- a public record containing only assistant-visible context and your request;
- a private record containing the complete scenario card;
- an audit record containing hashes and completion status.

The app makes no model or API call while you complete Phase 1.

## Start the app

From the repository root:

```bash
.venv/bin/python -m pip install -r requirements-pilot-ui.txt
./scripts/run-prospective-language-pilot.sh
```

Open the local address printed by Streamlit, normally `http://localhost:8501`. Use participant code `P001` unless the
study administrator gives you another code.

When all 16 records are complete, download the public, private, and audit files if you want a second copy. Then stop;
assistant processing is a later phase.

The research audit checks completeness, the frozen study hash, every locked-record hash, the public/private export
hashes, the absence of private scenario fields in the public file, and an assistant-generation count of exactly zero.

## Clarification batch

After the separate assistant batch and protocol review, the same app may display a clarification phase. It shows the
original scenario, your locked request, and the exact frozen assistant questions. Answer naturally from the scenario
card. It is acceptable to say that you do not know or to use the unable-to-answer option.

Complete all displayed clarifications before any terminal assistant continuation. Clarification responses are also
immutable. The app itself performs zero model generations while collecting them. When the completion screen appears,
stop again so the batch can be audited.
