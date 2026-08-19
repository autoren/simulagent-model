# V203 independent confirmation-feasibility results

## Result

V203 is a verified negative source-availability result. The exact frozen 14-contract task cannot be reconstructed on
SGD train/test alone, so no genuinely partition-independent confirmation population is selectable.

The train/test census covered 8 of 14 exact semantic contracts. Six had zero matching records and zero matching
partition-dialogue keys:

- `C_1b099ede5d276d0d7e686863` (`TransferMoney`);
- `C_4ab8adeabddfae28cdfd8586` (`CheckBalance`);
- `C_921e592a9360e0f6763b4faa` (`RentMovie`);
- `C_9e40b8c76ec62bd62e010c18` (`SearchOnewayFlight`);
- `C_b0fc70d7abe5fe162310f6cf` (`FindMovies`); and
- `C_c967f1c0f09afaef3cf4896e` (`SearchRoundtripFlights`).

Therefore exact contract coverage was `8` rather than `14`, minimum per-contract dialogue support was `0` rather
than `6`, and complete target expressibility failed. Exact singleton source-annotation mapping, provenance, allowed
partition, and prior-overlap gates passed but cannot compensate for missing targets.

## Diagnostic remainder

After excluding every V183 and V191 source record and partition-dialogue key, the mixed dev/train/test remainder did
cover all 14 contracts. Its limiting contract still had 47 unique dialogues. This does not rescue the confirmation
branch: V203 prospectively marked that family diagnostic-only because six contracts require reuse of the SGD
development partition. Reclassifying it after observing the favorable count would weaken the independence question.

The previously audited alternatives also remain ineligible. Taskmaster lacked deterministic turn-level active-intent
and state linkage under V87. The audited MultiWOZ reference lacked that linkage, a single machine-readable service
schema, and the frozen synthetic-family exclusion, and neither source has an exact mapping artifact for this
14-contract universe. Similar intent names are not exact semantic contracts.

## Boundary and decision

V203 read one pinned archive, three schema metadata files, one text-free candidate inventory, and two consumed-identity
artifacts. It emitted no utterance or dialogue text, inspected no protected language, scored no policy, loaded or ran
no model, and used no API, training, registration, mutation, service call, side effect, action, or execution.

The frozen decision is:

> `freeze_V203_negative_park_B1_confirmation_and_authorize_separate_richer_model_free_POMDP_preregistration_only`

V202 remains positive development evidence for top-3-plus-`OTHER` clarification, but no further language/model run is
justified from the available exact-contract sources. The next branch is the richer model-free POMDP specified in the
post-V202 roadmap: control-relevant semantic uncertainty, action-dependent sensing, delayed state-dependent loss, and
an exact oracle feasibility census before any candidate planner evaluation.
