from __future__ import annotations

import json

from cross_track_evidence_audit import run_and_write


def main() -> None:
    audit = run_and_write()
    reproducibility = audit["reproducibility_audit"]
    print(
        json.dumps(
            {
                "outcome_lock_count": reproducibility["outcome_lock_count"],
                "payload_valid_count": reproducibility["payload_valid_count"],
                "verified_dependency_pairs": reproducibility["verified_non_sensitive_dependency_pair_count"],
                "drifted_dependency_pairs": reproducibility["drifted_non_sensitive_dependency_pair_count"],
                "missing_dependency_pairs": reproducibility["missing_non_sensitive_dependency_pair_count"],
                "skipped_dependency_pairs": reproducibility["skipped_sensitive_or_nonlocal_dependency_pair_count"],
                "authorized_next_experiment_count": audit["stopping_decision"]["authorized_next_experiment_count"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
