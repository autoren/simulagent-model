from __future__ import annotations

import json

from model_free_reference_architecture import run_reference_architecture, write_bundle


def main() -> None:
    bundle = run_reference_architecture()
    write_bundle(bundle)
    print(
        json.dumps(
            {
                "passed": bundle["audit"]["passed"],
                "decision": bundle["audit"]["decision"],
                "trusted_route": bundle["result"]["typed_version_space"]["routed_decision"],
                "other_route": bundle["result"]["other_defer"]["decision"],
                "sandbox_disposition": bundle["result"]["reversible_sandbox"]["disposition"],
                "semantic_root_action": bundle["result"]["outside_semantic_terminal_planner"]["root_action"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
