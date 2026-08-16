#!/usr/bin/env python3
"""Retrieve pinned POBAX sources and build the sealed-input candidate bundle."""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT
from v62_external_pomdp import parse_pomdp_file, validate_model


def checkout_source(repository: str, commit: str, destination: Path) -> None:
    subprocess.run(["git", "init", "-q", str(destination)], check=True)
    subprocess.run(
        ["git", "-C", str(destination), "remote", "add", "origin", repository], check=True
    )
    subprocess.run(
        ["git", "-C", str(destination), "fetch", "-q", "--depth", "1", "origin", commit],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(destination), "checkout", "-q", "--detach", "FETCH_HEAD"],
        check=True,
    )


def canonical_model(model) -> dict[str, object]:
    return {
        "name": model.name,
        "states": list(model.states),
        "actions": list(model.actions),
        "observations": list(model.observations),
        "discount": model.discount,
        "initial": model.initial.tolist(),
        "transition": model.transition.tolist(),
        "observation": model.observation.tolist(),
        "reward": model.reward.tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--implementation-lock", default="configs/v62-implementation-lock.json")
    parser.add_argument("--bundle", default="data/v62-external-pomdp-transfer/bundle")
    args = parser.parse_args()
    implementation_path = (PROJECT_ROOT / args.implementation_lock).resolve()
    bundle = (PROJECT_ROOT / args.bundle).resolve()
    if bundle.exists():
        raise RuntimeError("V62 external bundle already exists")
    implementation = json.loads(implementation_path.read_text())
    if not implementation["authorization"]["build_and_audit_one_external_source_bundle"]:
        raise RuntimeError("V62 implementation lock does not authorize bundle construction")
    for path, digest in implementation["implementation_files_sha256"].items():
        if file_sha256(PROJECT_ROOT / path) != digest:
            raise RuntimeError(f"frozen V62 implementation changed: {path}")
    design_path = PROJECT_ROOT / implementation["design_lock"]
    design = json.loads(design_path.read_text())
    config = design["config_payload"]
    external = config["externalSource"]

    with tempfile.TemporaryDirectory(prefix="v62-bundle-source-") as temp_dir:
        checkout = Path(temp_dir) / "pobax"
        checkout_source(external["repository"], external["commit"], checkout)
        resolved_commit = subprocess.check_output(
            ["git", "-C", str(checkout), "rev-parse", "HEAD"], text=True
        ).strip()
        if resolved_commit != external["commit"]:
            raise RuntimeError("retrieved external commit does not match the design")
        for path, digest in external["files"].items():
            if file_sha256(checkout / path) != digest:
                raise RuntimeError(f"retrieved external source hash mismatch: {path}")
        if file_sha256(checkout / "LICENSE") != external["licenseSha256"]:
            raise RuntimeError("retrieved external license hash mismatch")

        source_root = bundle / "source"
        source_root.mkdir(parents=True)
        shutil.copy2(checkout / "LICENSE", source_root / "LICENSE")
        for path in external["files"]:
            destination = source_root / path
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(checkout / path, destination)

    source_metadata = {
        "repository": external["repository"],
        "commit": external["commit"],
        "license": external["license"],
        "license_sha256": external["licenseSha256"],
        "source_files_sha256": external["files"],
        "network_required_for_candidate_evaluation": False,
    }
    (bundle / "source-metadata.json").write_text(
        json.dumps(source_metadata, indent=2, sort_keys=True) + "\n"
    )

    model_summaries = {}
    for entry in config["benchmark"]["models"]:
        model_id = entry["id"]
        source_path = (
            bundle / "source" / f"pobax/envs/classic/POMDP/{model_id}.POMDP"
        )
        model = parse_pomdp_file(source_path)
        checks = validate_model(model)
        if not all(checks.values()):
            raise RuntimeError(f"candidate parser produced invalid model {model_id}: {checks}")
        model_path = bundle / "models" / model_id / "model.json"
        model_path.parent.mkdir(parents=True, exist_ok=True)
        model_path.write_text(json.dumps(canonical_model(model), indent=2, sort_keys=True) + "\n")
        model_summaries[model_id] = {
            "states": len(model.states),
            "actions": len(model.actions),
            "observations": len(model.observations),
            "discount": model.discount,
            "horizons": entry["horizons"],
            "validation": checks,
        }

    files = {}
    for path in sorted(item for item in bundle.rglob("*") if item.is_file()):
        relative = str(path.relative_to(bundle))
        files[relative] = {"sha256": file_sha256(path), "bytes": path.stat().st_size}
    manifest = {
        "schema_version": 62,
        "experiment": "v62_external_source_bundle",
        "implementation_lock": str(implementation_path.relative_to(PROJECT_ROOT)),
        "implementation_lock_sha256": file_sha256(implementation_path),
        "design_lock": str(design_path.relative_to(PROJECT_ROOT)),
        "design_lock_sha256": file_sha256(design_path),
        "external_commit": external["commit"],
        "source_model_count": 3,
        "task_cell_count": config["benchmark"]["taskCells"],
        "model_summaries": model_summaries,
        "files": files,
        "candidate_evaluations": 0,
        "human_record_access_count": 0,
        "model_forward_pass_count": 0,
    }
    manifest_path = bundle / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
