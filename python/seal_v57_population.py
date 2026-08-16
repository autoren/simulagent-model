#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json

from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--population", default="data/v57-definition-augmented-ontology-transfer")
    parser.add_argument("--audit", default="outputs/v57-definition-augmented-ontology-transfer/population-audit.json")
    parser.add_argument("--output", default="configs/v57-population-seal.json")
    args = parser.parse_args()
    population, audit_path, output = tuple(
        (PROJECT_ROOT / value).resolve()
        for value in (args.population, args.audit, args.output)
    )
    if output.exists():
        raise RuntimeError("V57 population already sealed")
    audit = json.loads(audit_path.read_text())
    manifest_path = population / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    lock_path = PROJECT_ROOT / manifest["implementation_lock"]
    if (
        not audit["passed"]
        or audit["manifest_sha256"] != file_sha256(manifest_path)
        or audit["implementation_lock_sha256"] != file_sha256(lock_path)
    ):
        raise RuntimeError("V57 population audit is not intact and bound")
    artifacts = {}
    for name, row in manifest["artifacts"].items():
        path = PROJECT_ROOT / row["path"]
        if file_sha256(path) != row["sha256"]:
            raise RuntimeError("V57 population changed after audit")
        artifacts[name] = row
    seal = {
        "schema_version": 57,
        "experiment": "v57_population_seal",
        "population": str(population.relative_to(PROJECT_ROOT)),
        "manifest": str(manifest_path.relative_to(PROJECT_ROOT)),
        "manifest_sha256": file_sha256(manifest_path),
        "artifacts": artifacts,
        "population_audit": str(audit_path.relative_to(PROJECT_ROOT)),
        "population_audit_sha256": file_sha256(audit_path),
        "implementation_lock": str(lock_path.relative_to(PROJECT_ROOT)),
        "implementation_lock_sha256": file_sha256(lock_path),
        "authorization": {
            "modify_v57_population": False,
            "write_and_audit_v57_candidate_runner": True,
            "run_v57_candidate_evaluation": False,
            "collect_human_language": False,
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
