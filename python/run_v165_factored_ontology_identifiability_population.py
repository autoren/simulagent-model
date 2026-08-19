#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT
from v165_factored_ontology_identifiability_population import (
    audit_population,
    build_population,
)


def payload_hash(value: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def main() -> None:
    lock_path = (
        PROJECT_ROOT
        / "configs/v165-factored-ontology-identifiability-population-lock.json"
    )
    output_root = (
        PROJECT_ROOT / "outputs/v165-factored-ontology-identifiability/population"
    )
    if output_root.exists():
        raise RuntimeError("V165 population may be built only once")
    lock = json.loads(lock_path.read_text())
    if payload_hash(
        {key: value for key, value in lock.items() if key != "lock_payload_sha256"}
    ) != lock["lock_payload_sha256"]:
        raise RuntimeError("V165 lock mismatch")
    dependency_keys = (
        "config",
        "parent_track_A_outcome",
        "roadmap",
        "plan",
        "protocol",
        "tests",
        "runner",
        "verifier",
        "auditor",
        "design_audit",
    )
    for key in dependency_keys:
        if file_sha256(PROJECT_ROOT / lock[key]) != lock[f"{key}_sha256"]:
            raise RuntimeError(f"V165 dependency drifted: {key}")
    config = lock["config_payload"]
    population = build_population(config)
    audit = audit_population(population, config)
    access = audit["access"]
    access["population_build_count"] = 1
    passed = audit["passed"]
    decision = (
        config["decisionRule"][
            "ifEveryPopulationAndIdentifiabilityGatePasses"
        ]
        if passed
        else config["decisionRule"]["otherwise"]
    )
    asset_paths = {
        "frozen_ontology": output_root / "frozen-ontology.json",
        "public_records": output_root / "public-records.json",
        "hidden_records": output_root / "hidden-records.json",
        "population_summary": output_root / "population-summary.json",
    }
    for key, path in asset_paths.items():
        write_json(path, population[key])
    output_integrity = {
        key: {
            "path": str(path.relative_to(PROJECT_ROOT)),
            "sha256": file_sha256(path),
        }
        for key, path in asset_paths.items()
    }
    result = {
        "schema_version": "165-factored-ontology-identifiability-population-result",
        "experiment": config["experiment"],
        "passed": passed,
        "decision": decision,
        "population_audit": audit,
        "output_integrity": output_integrity,
        "access": access,
        "claim_boundary": config["claimBoundary"],
    }
    write_json(output_root / "result.json", result)
    print(
        json.dumps(
            {
                "passed": passed,
                "decision": decision,
                "summary": audit["summary"],
                "version_space_size_by_status": audit[
                    "version_space_size_by_status"
                ],
                "checks": audit["checks"],
                "access": access,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
