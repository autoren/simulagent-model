# V43 post-hoc graph-metric diagnostic

This is a labeled post-hoc diagnostic. It does not revise V43's sealed failed qualification.

- Ordered-list exactness reproduced at `0.262`.
- Canonical row-set exactness was `1.000`.
- Semantic content mismatches: `0` of `1147` graphs.
- Ordering-only mismatches: `847`.

The reference rows were ordered using canonical entity IDs before those IDs were replaced with hashed aliases. Compiled rows were ordered after aliasing. Direct list equality therefore treated semantically identical graphs with different row order as unequal.

The proper next step is a separately preregistered measurement-repair confirmation over the immutable V43 artifacts. Only the graph comparator may change; every other V43 metric must reproduce exactly.
