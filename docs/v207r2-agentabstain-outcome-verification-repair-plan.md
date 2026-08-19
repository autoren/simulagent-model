# V207r2 AgentAbstain outcome-verification repair plan

## Failure isolated

The first V207r1 outcome verification reconstructed the pinned metadata census and result exactly. Every scientific, transport, access, and documentation check passed except design-dependency exactness because the living research roadmap had been updated after the roadmap file was included in the V207r1 lock.

## Repair

V207r2 is verification-only. The post-lock roadmap addition is removed so the roadmap again matches the V207r1 dependency hash. The failed audit is preserved. V207r2 reads only the already stored summary and result, deterministically rebuilds their audit, and verifies that the failed check set contained only dependency exactness. It performs no network request and does not rerun the scientific census or any model.

## Boundary

The V207r1 scientific result remains negative and its transport result remains positive. Source artifacts, task-language firewall, qualification rules, counts, decisions, and authorization do not change. Passing V207r2 authorizes only a subsequent roadmap update and separately preregistered F1 source-availability census.
