"""Materialize the V22r2 open relational-language grounding corpus."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from audit_v22_relational import read_records
from v10_protocol import file_sha256
from v22_relational import canonical_json
from v22r2_grounding import PROJECT_ROOT, build_corpus, sha256_text


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("".join(canonical_json(row) + "\n" for row in rows))


def corpus_hash(rows: list[dict[str, Any]]) -> str:
    return sha256_text("".join(canonical_json(row) + "\n" for row in rows))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/dataset.v22r2.json")
    args = parser.parse_args()
    config_path = (PROJECT_ROOT / args.config).resolve()
    config = json.loads(config_path.read_text())
    v22_config_path = PROJECT_ROOT / config["sourceV22Config"]
    v22_config = json.loads(v22_config_path.read_text())
    source_root = PROJECT_ROOT / config["sourceV22Dataset"]
    source_records = read_records(source_root)
    records, scenes = build_corpus(source_records, config, v22_config)
    output = PROJECT_ROOT / config["outputDir"]
    if output.exists():
        raise RuntimeError(f"V22r2 output already exists: {output}")
    (output / "records").mkdir(parents=True)
    (output / "scenes").mkdir(parents=True)
    for split in config["splits"]:
        write_jsonl(
            output / "records" / f"{split}.jsonl",
            [row for row in records if row["split"] == split],
        )
        write_jsonl(
            output / "scenes" / f"{split}.jsonl",
            [row for row in scenes if row["split"] == split],
        )
    ordered_records = sorted(records, key=lambda row: row["id"])
    ordered_scenes = sorted(scenes, key=lambda row: row["id"])
    manifest = {
        "schema_version": "22r2",
        "experiment": config["experiment"],
        "config": str(config_path.relative_to(PROJECT_ROOT)),
        "config_sha256": file_sha256(config_path),
        "source": {
            "v22_config": config["sourceV22Config"],
            "v22_config_sha256": file_sha256(v22_config_path),
            "v22_manifest": config["sourceV22Manifest"],
            "v22_manifest_sha256": file_sha256(PROJECT_ROOT / config["sourceV22Manifest"]),
            "v22_audit": config["sourceV22Audit"],
            "v22_audit_sha256": file_sha256(PROJECT_ROOT / config["sourceV22Audit"]),
            "v22_result": config["sourceV22Result"],
            "v22_result_sha256": file_sha256(PROJECT_ROOT / config["sourceV22Result"]),
        },
        "episodes": len(records),
        "scenes": len(scenes),
        "episode_split_counts": dict(sorted(Counter(row["split"] for row in records).items())),
        "scene_split_counts": dict(sorted(Counter(row["split"] for row in scenes).items())),
        "role_counts": dict(sorted(Counter(row["role"] for row in scenes).items())),
        "record_corpus_sha256": corpus_hash(ordered_records),
        "scene_corpus_sha256": corpus_hash(ordered_scenes),
        "new_model_forward_passes": 0,
        "model_predictions_read": 0,
        "adapter_training_runs": 0,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
