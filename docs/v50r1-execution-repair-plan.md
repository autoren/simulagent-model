# V50r1 execution repair

The first sealed V50 execution stopped before creating mechanic metrics or a result. A non-gating diagnostic converted an extremely small positive `Decimal` posterior weight to binary float zero, then evaluated `log(0)`. This raised `ValueError` while assembling the first mechanic record.

V50r1 preserves the failed attempt and reuses the identical V50 implementation lock, corpus seal, populations, held-out outcomes, posterior/predictive calculations, metrics, gates, and decision hierarchy. The only permitted change is the effective-program-count diagnostic: entropy is evaluated in `Decimal`, exponentiated in `Decimal`, and converted to float only after the result is in the safe interval `[1, 48]`.

No scientific result from V50 was available when this repair was specified. V50r1 receives one execution authorization. It remains non-final and has no model access, training, active selection, language expansion, or evaluation-set selection.
