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
        "--audit",
        default="outputs/v55r1-delayed-consequence-adequacy-confirmation/evaluation-implementation-audit.json",
    )
    parser.add_argument(
        "--population-seal", default="configs/v55r1-population-seal.json"
    )
    parser.add_argument(
        "--output", default="configs/v55r1-evaluation-implementation-lock.json"
    )
    args = parser.parse_args()
    audit_path, seal_path, output = tuple(
        (PROJECT_ROOT / value).resolve()
        for value in (args.audit, args.population_seal, args.output)
    )
    if output.exists():
        raise RuntimeError("V55r1 evaluation implementation already frozen")
    audit = json.loads(audit_path.read_text())
    if (
        not audit["passed"]
        or audit["population_seal_sha256"] != file_sha256(seal_path)
        or any(
            file_sha256(PROJECT_ROOT / path) != digest
            for section in ("evaluation_files_sha256", "frozen_dependencies_sha256")
            for path, digest in audit[section].items()
        )
    ):
        raise RuntimeError("V55r1 evaluation audit is not intact and bound")
    lock = {
        "schema_version": 55,
        "revision": "r1",
        "experiment": "v55r1_evaluation_implementation_lock",
        "population_seal": str(seal_path.relative_to(PROJECT_ROOT)),
        "population_seal_sha256": file_sha256(seal_path),
        "evaluation_implementation_audit": str(audit_path.relative_to(PROJECT_ROOT)),
        "evaluation_implementation_audit_sha256": file_sha256(audit_path),
        "evaluation_files_sha256": audit["evaluation_files_sha256"],
        "frozen_dependencies_sha256": audit["frozen_dependencies_sha256"],
        "authorization": {
            "run_one_v55r1_evaluation": True,
            "run_additional_v55r1_evaluation": False,
            "run_additional_v55_evaluation": False,
            "change_evaluation_implementation": False,
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
