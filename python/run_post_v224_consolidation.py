from __future__ import annotations

import json

from post_v224_consolidation import build_consolidation, write_bundle


def main() -> None:
    bundle = build_consolidation()
    write_bundle(bundle)
    result = bundle["result"]
    print(
        json.dumps(
            {
                "passed": result["passed"],
                "gate_count": len(result["gates"]),
                "drift_count": result["dependency_drift"]["finding_count"],
                "architecture_component_count": len(result["reference_architecture"]["component_ids"]),
                "historical_roadmap_count": result["navigation"]["historical_roadmap_count"],
                "decision": result["decision"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
