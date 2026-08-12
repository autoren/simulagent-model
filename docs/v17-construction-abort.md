# V17 initial construction abort

The first locked V17 constructor was invoked once and stopped before creating `data/v17-final`. Its preregistered assertion required eight distinct simulator transitions, but the simulator's normalized `actualOutcome` plus `actionSurfaceDelta` yielded one code for a read-only inspection action.

Observed information was limited to the failed invariant `transition codes = 1`; no transition text, V17 record, label artifact, feature, prediction, or model score was written or read. The aborted construction-lock file hash is `75005d6b1539e35115fc102fa67dfa9e5dc66dce0753c487d04bc0f295c0b446`, and its config hash is `2d85f9a3a192df5ec8ca0722b3dbc5cd6b9638041540830d05c47a55ca4c0fba`.

Because the failure occurred before final-data construction, a fresh pre-data amendment may correct the transition identity. The aborted lock authorizes no further action and is retained at `configs/v17-aborted-construction-lock.json`.
