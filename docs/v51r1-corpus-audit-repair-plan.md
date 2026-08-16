# V51r1 preregistration: corpus identity audit repair

V51 corpus construction was frozen before generation. The generated corpus exactly matched the precommitted SHA-256 and passed the substantive pre-run checks: all 2,048 replications were present, every target support likelihood was positive, every simulated query configuration was retained, all prior programs were represented, observation schemas were valid, and value-independent designs reproduced.

The corpus audit nevertheless stopped before sealing because its `structural_key` identity used only entities, initial state, and actions. In a partial-observation experiment the known observation-mask schedule is part of the experimental design. Omitting it aliases distinct experiments and can create false duplicate and historical-overlap findings.

V51r1 authorizes one measurement repair: case identity becomes the canonical hash of entities, initial state, actions, and masks. The unchanged V51 corpus may be sealed only if the original audit has no failure other than the identity firewall, every other source check passed, every V51 support and query observation design is unique in its split, the splits are disjoint, and their union has zero overlap with V50 under the same complete identity.

V51r1 does not authorize corpus regeneration or byte changes, changes to seeds or programs, inference changes, rank or control changes, gate changes, calibration-outcome access, model access, or more than the original single calibration run. The repair is therefore incapable of selecting a favorable SBC result.
