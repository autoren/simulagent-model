# Dataset audit

Dataset: `data/full`

## Corpus

- Records: 10,038
- Scenarios: 31
- Split groups: 26
- Reachable states: 1,240
- State-changing targets: 56.6%
- Successful actions: 94.0%

## Split integrity

- Groups crossing splits: 0
- Scenarios crossing splits: 0
- Duplicate record IDs: 0

## Observational signatures

| Signature | Unique | Ambiguous groups | Ambiguous records | Cross-split overlaps | Exact-match ceiling |
| --- | ---: | ---: | ---: | ---: | ---: |
| exact_agent_prompt | 1,525 | 612 | 7,763 | 581 | 76.56% |
| exact_privileged_prompt | 2,829 | 933 | 5,369 | 847 | 83.82% |
| observation_action | 1,525 | 612 | 7,763 | 581 | 76.56% |
| lossy_structural_observation_action | 440 | 301 | 9,258 | 257 | 73.20% |

The structural signature intentionally removes prose, beliefs, memories, and history. Its ceiling
measures how much those omitted features matter; it is not an identifiability claim. The exact-agent
signature is the relevant check for contradictory supervised examples.

For the exact agent prompt, the audit also reports how many test prompts already occur in
training. High coverage here means scenario-level splitting does not produce input-level novelty.

## Test transfer diagnostic

| Track | Test records seen verbatim in train | Train-majority exact match |
| --- | ---: | ---: |
| Agent | 100.00% | 84.56% |
| Privileged snapshot | 100.00% | 90.76% |

The current privileged snapshot includes flags, rooms, inventory, and scalar state, but not the
scenario's transition rules. Its remaining contradictory labels show that it is not yet a complete
Markov-state representation.

## Action distribution

| Action | Records |
| --- | ---: |
| inspect | 4,809 |
| move | 2,815 |
| take | 797 |
| talk | 292 |
| use | 85 |
| wait | 1,240 |
