# Dataset v2 audit

## Gate results

| Check | Result |
| --- | ---: |
| Agent unique prompts | 1,525 / 1,525 |
| Agent exact prompts crossing splits | 0 |
| Agent test prompts seen in training | 0.00% |
| Agent identifiable prompts | 59.87% |
| Privileged contradictory prompts | 0 |
| Privileged exact prompts crossing splits | 0 |
| Privileged test prompts seen in training | 0.00% |

Agent ambiguity is now represented as a target property rather than contradictory rows.
The privileged track contains explicit transition rules and is empirically Markov-complete
for the current target schema. Both tracks use prompt-disjoint context splits.

## Agent identifiability by split

| Split | Prompts | Identifiable | Ambiguous | Ambiguous rate |
| --- | ---: | ---: | ---: | ---: |
| train | 1,223 | 745 | 478 | 39.08% |
| valid | 145 | 52 | 93 | 64.14% |
| test | 157 | 116 | 41 | 26.11% |

## Agent possible-outcome counts

| Outcomes | Prompts |
| ---: | ---: |
| 1 | 913 |
| 2 | 478 |
| 3 | 101 |
| 4 | 19 |
| 5 | 14 |

## Fields varying in ambiguous prompts

| Field | Prompt groups |
| --- | ---: |
| environment_changed | 610 |
| flags_changed | 268 |
| visible_actions_added | 54 |
| hidden_actions_concealed | 54 |
| blocked_actions_added | 21 |
