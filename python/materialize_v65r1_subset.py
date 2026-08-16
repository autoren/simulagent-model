#!/usr/bin/env python3
"""Materialize the prospectively hash-selected public V65r1 subset."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT


def canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def subset_digest(seed: int, record_id: str) -> str:
    return hashlib.sha256(f"v65|subset|{seed}|{record_id}".encode()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lock", default="configs/v65r1-implementation-lock.json")
    parser.add_argument("--output", default="data/v65-smc2-eig-portability")
    args = parser.parse_args()
    lock_path = (PROJECT_ROOT / args.lock).resolve()
    output = (PROJECT_ROOT / args.output).resolve()
    if output.exists():
        raise RuntimeError("V65r1 subset directory already exists")
    lock = json.loads(lock_path.read_text())
    if not lock["authorization"]["materialize_and_audit_subset"]:
        raise RuntimeError("V65r1 implementation lock does not authorize subset materialization")
    for relative, digest in lock["source_sha256"].items():
        if file_sha256(PROJECT_ROOT / relative) != digest:
            raise RuntimeError(f"frozen V65r1 implementation changed: {relative}")
    design_path = (PROJECT_ROOT / lock["design_lock"]).resolve()
    if file_sha256(design_path) != lock["design_lock_sha256"]:
        raise RuntimeError("V65r1 design lock changed after implementation freeze")
    design = json.loads(design_path.read_text())
    config = design["config_payload"]
    source_path = (PROJECT_ROOT / config["pairedReuseBoundary"]["sourcePopulation"]).resolve()
    expected_source_hash = config["pairedReuseBoundary"]["sourcePopulationSha256"]
    if file_sha256(source_path) != expected_source_hash:
        raise RuntimeError("sealed V64 public selection file changed")
    source = read_jsonl(source_path)
    if len(source) != config["pairedReuseBoundary"]["sourceRecords"]:
        raise RuntimeError("sealed V64 public selection count changed")

    allowed = set(config["subset"]["publicFieldsOnly"])
    if any(set(row) != allowed for row in source):
        raise PermissionError("V64 source public file contains undeclared fields")
    seed = int(config["subset"]["selectionSeed"])
    selected: list[dict] = []
    provenance: list[dict] = []
    for prefix in config["subset"]["prefixLengths"]:
        stratum = [row for row in source if int(row["prefix_length"]) == int(prefix)]
        if len(stratum) != config["pairedReuseBoundary"]["sourceRecordsPerPrefixLength"]:
            raise RuntimeError("V64 public prefix stratum count changed")
        ranked = sorted(
            stratum,
            key=lambda row: (subset_digest(seed, str(row["record_id"])), str(row["record_id"])),
        )
        kept = ranked[: int(config["subset"]["recordsPerPrefixLength"])]
        selected.extend(kept)
        provenance.extend(
            {
                "record_id": row["record_id"],
                "prefix_length": prefix,
                "selection_digest": subset_digest(seed, str(row["record_id"])),
                "rank_within_prefix": rank,
            }
            for rank, row in enumerate(kept)
        )
    if len(selected) != config["subset"]["records"]:
        raise RuntimeError("V65r1 subset size does not match preregistration")

    output.mkdir(parents=True)
    subset_path = output / "subset-public.jsonl"
    provenance_path = output / "subset-provenance.jsonl"
    subset_path.write_text("".join(canonical_json(row) + "\n" for row in selected))
    provenance_path.write_text("".join(canonical_json(row) + "\n" for row in provenance))
    manifest = {
        "schema_version": "65r1",
        "experiment": "v65r1_subset_manifest",
        "implementation_lock": str(lock_path.relative_to(PROJECT_ROOT)),
        "implementation_lock_sha256": file_sha256(lock_path),
        "materializer": "python/materialize_v65r1_subset.py",
        "materializer_sha256": file_sha256(Path(__file__).resolve()),
        "source_public": str(source_path.relative_to(PROJECT_ROOT)),
        "source_public_sha256": file_sha256(source_path),
        "source_records_loaded": len(source),
        "source_selection_audit_records_loaded": 0,
        "source_evaluation_records_loaded": 0,
        "selection_seed": seed,
        "selection_rule": config["subset"]["selectionRule"],
        "counts": {
            "subset_public": len(selected),
            "subset_provenance": len(provenance),
        },
        "files": {
            "subset_public": {
                "path": str(subset_path.relative_to(PROJECT_ROOT)),
                "sha256": file_sha256(subset_path),
            },
            "subset_provenance": {
                "path": str(provenance_path.relative_to(PROJECT_ROOT)),
                "sha256": file_sha256(provenance_path),
            },
        },
    }
    manifest_path = output / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
