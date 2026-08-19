# V172 trusted shadow integration population results

## Outcome

V172 passed every population gate without running an integration policy or sandbox transaction. Complete
enumeration produced 448 unique three-constraint truth-table states, and every state was retained with an exact
32-candidate version space.

Frozen class coverage was:

| Expressibility classes present | State count |
|---:|---:|
| 1 | 212 |
| 2 | 104 |
| 3 | 132 |

The 132 three-class states are structurally eligible for the unchanged V167 class-balanced prior. All 316
ineligible states remain in the population. Every candidate in every eligible version space became a target case,
yielding 4,224 frozen cases: 144 alias, 228 composition, and 3,852 provisional-primitive candidates. No target was
subsampled. Each target's exact weight assigns one third of prior mass to its class and divides that mass uniformly
within the state and class.

State membership SHA-256 is `c71b1517bdc05aa09443bdc9e376d3e51474e8013bbdefc81e05988c3af5a7a5`.
Target membership SHA-256 is `181d0019981467df1b445bea6044a346c7a44abfe3be0fa8226646785c314ffd`.

## Interpretation and boundary

V172 cleanly separates population construction from integration scoring. Eligibility and target identities use
only exact version-space membership and frozen class metadata. They do not use planner decisions, value,
transaction success, or sandbox outcomes. The large raw provisional count will not silently dominate the policy
estimand because the preregistered target weights reproduce V167's class-balanced prior; raw class-conditional
counts remain available as a separate descriptive view.

The population is project-authored and procedural, not external or human-authored. No language, model, API,
training, policy score, sandbox transaction, ontology registration, trusted-state mutation, real service, side
effect, or execution occurred.

Freeze V172 and authorize only a separately preregistered trusted-only shadow integration over all 132 eligible
states and all 4,224 target cases. The population may not be selected, subsampled, or tuned after integration
outcomes are opened.
