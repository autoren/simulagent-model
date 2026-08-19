# V224r2 two-stage query-execution repair plan

V224r2 implements the frozen preliminary-then-deep sequence explicitly:

1. enumerate all records using the same 48 search slices and only issue, actor, label, state, duplicate, and closing-PR
   count metadata;
2. apply the unchanged preliminary mappings and hash selection;
3. stop immediately if any preliminary stratum is below 24; and
4. only if that gate passes, fetch the original full safe node selection for chosen accepted records and duplicate
   canonicals, followed by the unchanged pull-file and release-ID provenance checks.

The thin query selects a strict subset of fields from the original safe query. The deep query is byte-identical to the
original locked node query. Neither selects titles, bodies, comments, review text, or commit messages. The source
window, exclusions, outcomes, human-independence rules, AI exclusions, sample seed, sample cap, thresholds, and branch
rule do not change.

The transport retries HTTP 502/503/504 at most three times for the exact same page. Retries do not alter or tune the
data. V224r2 authorizes one completed metadata census and records both prior failed implementation attempts as
non-scientific provenance.

