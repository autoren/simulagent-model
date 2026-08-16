# V58 pilot release gate

No pilot packet may be rendered, shared, or collected until a real collection coordinator supplies a declaration conforming to `configs/v58-pilot-coordinator-declaration.schema.json` and the frozen release-gate implementation accepts it.

The declaration uses pseudonymous role IDs only. The coordinator keeps the actual identity map, consent records, and collection-storage description outside this repository and binds each external record by SHA-256. Exactly two real pilot writers must be mapped to the two sealed pilot packet IDs. At least two validators, one adjudicator, and one candidate-developer role must be declared. Writer, validator, adjudicator, coordinator, and candidate roles must be disjoint where required.

All ten declaration attestations are noncompensatory. In particular, the project owner must explicitly authorize pilot release only; evaluation packets remain unreleased, candidate development remains frozen, and generative writing assistance remains prohibited. A malformed declaration cannot be repaired silently after collection starts: it must be replaced and re-audited before release.

When valid, `python/freeze_v58_pilot_release.py` creates a release lock authorizing only the two sealed pilot packets and the already frozen offline renderer/intake tools. It does not authorize evaluation collection, candidate development, or any language claim.
