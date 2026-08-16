#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json

from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--implementation-lock", default="configs/v53r2-implementation-lock.json")
    parser.add_argument(
        "--audit", default="outputs/v53r2-continuous-parameter-smc2/population-audit.json"
    )
    parser.add_argument("--output", default="configs/v53r2-population-seal.json")
    args = parser.parse_args()
    lock_path, audit_path, output = tuple(
        (PROJECT_ROOT / value).resolve()
        for value in (args.implementation_lock, args.audit, args.output)
    )
    if output.exists():
        raise RuntimeError("V53r2 populations already sealed")
    audit = json.loads(audit_path.read_text())
    if not audit["passed"] or audit["implementation_lock_sha256"] != file_sha256(lock_path):
        raise RuntimeError("V53r2 population audit is not bound to implementation")
    root = PROJECT_ROOT / "data/v53r2-continuous-parameter-smc2"
    manifest = root / "manifest.json"
    populations = {}
    for name in ("exact", "sbc", "scale"):
        path = root / f"{name}.jsonl"
        populations[name] = {
            "path": str(path.relative_to(PROJECT_ROOT)),
            "sha256": file_sha256(path),
            "records": len(path.read_text().splitlines()),
        }
    seal = {
        "schema_version": 53,
        "revision": "r2",
        "experiment": "v53r2_population_seal",
        "implementation_lock": str(lock_path.relative_to(PROJECT_ROOT)),
        "implementation_lock_sha256": file_sha256(lock_path),
        "population_audit": str(audit_path.relative_to(PROJECT_ROOT)),
        "population_audit_sha256": file_sha256(audit_path),
        "manifest": str(manifest.relative_to(PROJECT_ROOT)),
        "manifest_sha256": file_sha256(manifest),
        "populations": populations,
        "authorization": {
            "run_v53r2_evaluation_once": True,
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
