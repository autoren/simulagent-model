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
        default="outputs/v55r1-delayed-consequence-adequacy-confirmation/population-audit.json",
    )
    parser.add_argument("--output", default="configs/v55r1-population-seal.json")
    args = parser.parse_args()
    audit_path = (PROJECT_ROOT / args.audit).resolve()
    output = (PROJECT_ROOT / args.output).resolve()
    if output.exists():
        raise RuntimeError("V55r1 population already sealed")
    audit = json.loads(audit_path.read_text())
    population_path = PROJECT_ROOT / audit["population"]
    manifest_path = PROJECT_ROOT / audit["manifest"]
    lock_path = PROJECT_ROOT / audit["implementation_lock"]
    if (
        not audit["passed"]
        or audit["population_sha256"] != file_sha256(population_path)
        or audit["manifest_sha256"] != file_sha256(manifest_path)
        or audit["implementation_lock_sha256"] != file_sha256(lock_path)
    ):
        raise RuntimeError("V55r1 population audit is not intact and bound")
    seal = {
        "schema_version": 55,
        "revision": "r1",
        "experiment": "v55r1_population_seal",
        "population": audit["population"],
        "population_sha256": audit["population_sha256"],
        "manifest": audit["manifest"],
        "manifest_sha256": audit["manifest_sha256"],
        "population_audit": str(audit_path.relative_to(PROJECT_ROOT)),
        "population_audit_sha256": file_sha256(audit_path),
        "implementation_lock": audit["implementation_lock"],
        "implementation_lock_sha256": audit["implementation_lock_sha256"],
        "authorization": {
            "write_and_audit_v55r1_evaluation_implementation": True,
            "run_v55r1_evaluation": False,
            "change_population": False,
            "rerun_v55_population": False,
            "preregister_formal_verification": False,
            "run_formal_verification": False,
            "language_grounding": False,
            "model_access": False,
        },
    }
    seal["seal_payload_sha256"] = hashlib.sha256(
        json.dumps(seal, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    output.write_text(json.dumps(seal, indent=2, sort_keys=True) + "\n")
    print(json.dumps(seal, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
