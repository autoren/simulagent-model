# V39 results: declared-language compiler

Decision: `declared_language_compiler_pass_preregister_fresh_confirmation`.

V39 tests exact compilation within an explicitly declared controlled language. It does not test open-ended paraphrase understanding.

| Primary metric | Result |
|---|---:|
| Supported coverage | 1.000 |
| Supported exact parse | 1.000 |
| Supported compiled truth | 1.000 |
| Worst held-out composition cell | 1.000 |
| Malformed-input abstention | 1.000 |
| Unknown-lexeme abstention | 1.000 |
| Ambiguity safety | 1.000 |

All preregistered gates passed: `true`.
The non-gating novel-paraphrase exact parse rate was 0.000; this records the declared scope boundary.

Interpretation: the remaining V38 failure was an interface problem inside the tested controlled-language scope. V39 shows that once predicate and operator language are both declared, the existing symbolic representation can receive exact, safely compiled semantics. The next claim still requires a fresh preregistered confirmation population.

Post-result integrity audit: `pass`.
