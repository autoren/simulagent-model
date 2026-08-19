# V101 MASSIVE Population Result

## Outcome

V101 passed every locked population gate. It selected 512 identifiers exclusively from the frozen,
text-free V100 candidate index: 256 from MASSIVE validation for development and 256 from MASSIVE test
as the protected evaluation set. The two populations are identifier-disjoint.

Each role contains exactly 64 records from each source class:

| Role | Familiar known | Unfamiliar known | Novel valid | Unsupported | Total |
|---|---:|---:|---:|---:|---:|
| Development | 64 | 64 | 64 | 64 | 256 |
| Protected test | 64 | 64 | 64 | 64 | 256 |

Both known classes cover all three catalog scenarios in each role, novel-valid covers both hidden-intent
scenarios, and unsupported covers the one withheld scenario. Development intent coverage is 8, 10, 2,
and 4 for the four classes respectively; protected-test coverage is 7, 10, 2, and 4. Every locked
minimum passed.

The selected-population payload SHA-256 is
`284e7d464967a385f711e10c8919a31dc4c36f7ab288dd99721d0b96dcc7d8dc`.

## Boundary

V101 read the text-free source inventory once. It did not reopen the MASSIVE archive, extract or emit a
selected utterance, inspect language manually, load or run a model, call an LLM API, train an adapter,
call a real service, or cause an external side effect.

This is a population-selection result only. It authorizes preregistration of an automatic selected-
language extraction stage; it does not authorize extraction immediately. The protected test must remain
closed until prompt, output contract, deterministic controls, metrics, and pass/fail gates are frozen.
