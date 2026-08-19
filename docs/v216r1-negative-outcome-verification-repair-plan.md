# V216r1 negative-outcome verification repair plan

## Defect

The V216 design explicitly specifies a positive and a negative branch, but its outcome verifier includes a
positive-only condition: it requires the reconstructed scientific audit itself to pass. That condition prevents the
verifier from freezing a faithful negative result.

## Scope

V216r1 may only:

1. hash and validate the existing V216 design lock, raw payloads, derived artifacts, summary, result, and results
   document;
2. reconstruct the exact V216 metrics and scientific audit;
3. prove that the sole failed scientific check, branch, decision, counts, and access boundaries are unchanged; and
4. freeze an outcome lock whose repair audit passes while whose embedded V216 scientific outcome remains negative.

It may not rerun retrieval, rebuild the population, modify a V216 file, change the 20,000-term gate, reinterpret V216
as positive, authorize V217, open the protected partition for a method, or run a model.

## Decision

A passing repair authorizes selection of a post-V216 roadmap that does not rely on V217 authorization. Failure rejects
the repair and leaves V216 unchanged.

