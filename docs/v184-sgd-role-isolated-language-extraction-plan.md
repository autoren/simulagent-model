# V184 SGD role-isolated language extraction plan

## Purpose

V184 performs the one exact language extraction authorized by V183. It writes development and protected SGD conversations to separate immutable artifacts and extracts the six declared known schemas into a third artifact. It does not evaluate language.

## Observable record

Each record contains only its opaque V183 identifier, role, observation-presence flag, fallible presented candidate, and the conversation prefix through the selected user turn. Conversation turns contain only speaker and utterance.

Gold service, intent, domain, source identifier, truth kind, contract identity, compatibility set, evidence status, evaluation choice, frames, actions, state, slot values, and spans are forbidden. Missing controls have no conversation.

## Declared catalog

The catalog contains exactly the six frozen declared known choices and their source-authored service, intent, and slot descriptions. Each choice is linked to its V183 semantic contract hash. Provisional and unsupported schema descriptions remain hidden; the model cannot solve novelty by selecting from a leaked list of withheld answers.

The catalog is descriptive evidence, not executable authority. It cannot register or invoke a capability.

## Isolation and access

Development and protected artifacts must reconstruct the 132/132 V183 role split exactly, including 120 source conversations and 12 missing controls in each role. The artifacts must have zero opaque-identifier and source-identifier overlap.

The extraction is automatic. Neither role's language is printed or manually inspected. Protected text may be written and independently hash-verified, but it may not be read during later development. No policy, model, API, training, ontology registration, state mutation, service, action, side effect, or execution is allowed.

## Decision

Passing authorizes only prospective design of the deterministic observable interface, controls, metrics, and utterance-level evidence-sufficiency evaluator. It does not authorize scoring development text, opening protected text, or running a local/API model.
