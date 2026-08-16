#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json

from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/v55r1-delayed-consequence-adequacy-confirmation.json",
    )
    parser.add_argument(
        "--plan",
        default="docs/v55r1-delayed-consequence-adequacy-confirmation-plan.md",
    )
    parser.add_argument(
        "--audit",
        default="outputs/v55r1-delayed-consequence-adequacy-confirmation/design-audit.json",
    )
    parser.add_argument("--output", default="configs/v55r1-design-lock.json")
    args = parser.parse_args()
    config_path, plan_path, audit_path, output = tuple(
        (PROJECT_ROOT / value).resolve()
        for value in (args.config, args.plan, args.audit, args.output)
    )
    if output.exists():
        raise RuntimeError("V55r1 design already frozen")
    audit = json.loads(audit_path.read_text())
    source_paths = [
        PROJECT_ROOT / audit[key]
        for key in (
            "source_outcome_lock", "source_localization",
            "source_implementation_lock",
        )
    ]
    if (
        not audit["passed"]
        or audit["config_sha256"] != file_sha256(config_path)
        or audit["preregistration_sha256"] != file_sha256(plan_path)
        or any(
            audit[key] != file_sha256(path)
            for key, path in zip(
                (
                    "source_outcome_lock_sha256",
                    "source_localization_sha256",
                    "source_implementation_lock_sha256",
                ),
                source_paths,
                strict=True,
            )
        )
    ):
        raise RuntimeError("V55r1 design audit is not intact and bound")
    config = json.loads(config_path.read_text())
    lock = {
        "schema_version": 55,
        "revision": "r1",
        "experiment": "v55r1_design_lock",
        "config": str(config_path.relative_to(PROJECT_ROOT)),
        "config_sha256": file_sha256(config_path),
        "preregistration": str(plan_path.relative_to(PROJECT_ROOT)),
        "preregistration_sha256": file_sha256(plan_path),
        "design_audit": str(audit_path.relative_to(PROJECT_ROOT)),
        "design_audit_sha256": file_sha256(audit_path),
        "source_outcome_lock": audit["source_outcome_lock"],
        "source_outcome_lock_sha256": audit["source_outcome_lock_sha256"],
        "source_localization": audit["source_localization"],
        "source_localization_sha256": audit["source_localization_sha256"],
        "source_implementation_lock": audit["source_implementation_lock"],
        "source_implementation_lock_sha256": audit[
            "source_implementation_lock_sha256"
        ],
        "config_payload": config,
        "authorization": {
            "write_and_audit_v55r1_implementation": True,
            "construct_v55r1_population": False,
            "run_v55r1_evaluation": False,
            "preregister_formal_verification": False,
            "run_formal_verification": False,
            "language_grounding": False,
            "model_access": False,
        },
    }
    lock["lock_payload_sha256"] = hashlib.sha256(
        json.dumps(lock, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    output.write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n")
    print(json.dumps(lock, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
