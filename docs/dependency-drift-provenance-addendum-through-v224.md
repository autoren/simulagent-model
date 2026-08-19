# Dependency-drift provenance addendum through V224

## Decision

None of the eight originally hashed dependency byte sequences is recoverable from any reachable ref or reflog. Each affected file appears in one reachable commit, already at its current hash, and every current file matches HEAD/index. Reconstructing an alleged original would therefore be guesswork. The files and historical locks remain unchanged; this append-only addendum preserves both hashes and narrows the reconstruction claim.

The outcome-lock payloads themselves remain valid. A drifted verifier/auditor means its exact historical executable is unavailable; a drifted results document means the present prose is not the exact document frozen by that outcome. Hash-matched result, summary, audit, access, and design artifacts listed in the machine-readable addendum remain the appropriate support where available.

## Findings

| Outcome lock | Dependency | Type | Frozen hash | Current hash | Recoverable |
|---|---|---|---|---|---|
| `configs/v78-clarification-outcome-lock.json` | `python/verify_and_freeze_v78_clarification_outcome.py` | executable_outcome_verifier | `14da46bfadbee3a8aacacd44a9929125bbd04abf0288ed467aafbfe4a4143f89` | `5f9ca5f89bbcfadbcd96856b8e80bada7979a68b0394fd2c5a12308251cf650f` | no |
| `configs/v79-terminal-utility-outcome-lock.json` | `python/verify_and_freeze_v79_terminal_utility_outcome.py` | executable_outcome_verifier | `ad12eceada068fbd43e2a4bdc63b52cb79402c3c397384270a28f454881ae3e2` | `abd1ce35419006d118dee42098cda065c7218f56efff9d86a5f91d6e79d61162` | no |
| `configs/v94-global-open-set-source-outcome-lock.json` | `docs/v94-global-open-set-source-results.md` | narrative_results_document | `6650a3e7e0b4549e336e75a1119b2f2b707f33632befe28ca6a0c627ae69bb90` | `cf79b33ab92551e04503bf600e31398d90d8b10e6e9a1747b13ccd5c15bba889` | no |
| `configs/v106-open-world-development-benchmark-technical-outcome-lock.json` | `python/audit_and_freeze_v106_development_benchmark.py` | executable_auditor | `d750e24ae7036957e6149915b7f6bc1c157763f7ee796b6a8df7db7978456f7d` | `c0f5227a51887a2d0bea37f6b5dae2c341b148863b99dca8da99bbb4039c1ce0` | no |
| `configs/v118-evidence-identifiability-audit-outcome-lock.json` | `docs/v118-evidence-identifiability-audit-results.md` | narrative_results_document | `9a1c37ffdd05bf1031c2008b64da9e1054fc404eeaada8ffc85d4a00ee2ceec5` | `ad00a90ef82a4c914d9fe3711457625b126962e77a733a41eccb928c456a3cb5` | no |
| `configs/v119-asymmetric-adaptive-evidence-outcome-lock.json` | `docs/v119-asymmetric-adaptive-evidence-results.md` | narrative_results_document | `4727bc16c7b1ac309be195826cd5a464daa27d0d183943a686296830130f7523` | `7baab03d1967531c2b8ab436a7e87c67bb3684e44c7bc460a22a738facc12f52` | no |
| `configs/v120-selective-query-value-audit-outcome-lock.json` | `docs/v120-selective-query-value-audit-results.md` | narrative_results_document | `ef975aa03573c1461335a5e704d3b0d323255ce703bccc5e0facbea80a331984` | `cb3ce30602f9be3f0232b80e06bc6ce448f06e32f5e49196e8e49ba9b96ec0ee` | no |
| `configs/v121-prequery-value-selectivity-envelope-outcome-lock.json` | `docs/v121-prequery-value-selectivity-envelope-results.md` | narrative_results_document | `53cc34691cee7420a5cdcd88f09e38d6dd362c604228ff8e4429e7d4366a6c45` | `2d20e5326d83f53b6593394cb11bca0a0245a9f1c664518c534ee7484450ad06` | no |

## Interpretation

- V78 and V79 have executable-verifier provenance gaps.
- V106 has an executable-auditor provenance gap for the failed-auditor dependency.
- V94 and V118-V121 have narrative-results-document provenance gaps.
- These are not evidence that the outcome value changed, but they prevent claiming exact reconstruction through the drifted dependency.
- Each current file is byte-identical to its sole reachable/reflog version (zero changed bytes and lines); the unavailable frozen bytes cannot be directly diffed.
- Repair requires an exact content-addressed copy whose SHA-256 matches the frozen hash. Editing the current file until a plausible resemblance is reached is forbidden.

The complete commit inventory and surviving hash-matched support are in `outputs/cross-track-evidence-audit-through-v224/dependency-drift-provenance-addendum.json`.
