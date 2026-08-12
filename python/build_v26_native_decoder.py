"""Build the immutable one-prompt-per-evidence V26 corpus."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from typing import Any

from audit_v22r2_grounding import read_jsonl_directory
from v10_protocol import file_sha256
from v22_relational import canonical_json
from v22r2_grounding import PROJECT_ROOT
from v24_cross_encoder import sha256_text


def build_rows(config: dict[str, Any]) -> list[dict[str, Any]]:
    v25_lock = json.loads((PROJECT_ROOT / config["sourceV25Lock"]).read_text())
    v24_lock = json.loads((PROJECT_ROOT / v25_lock["source"]["v24_lock"]).read_text())
    original_lock = json.loads((PROJECT_ROOT / v24_lock["source"]["v22r2_lock"]).read_text())
    scenes = read_jsonl_directory(PROJECT_ROOT / original_lock["source"]["dataset"] / "scenes")
    predictions = {
        row["scene_id"]: row for row in [
            json.loads(line)
            for line in (PROJECT_ROOT / config["sourceV24Predictions"]).read_text().splitlines()
            if line.strip()
        ]
    }
    rows = []
    for scene in scenes:
        public = scene["agent_input"]
        evidence_text = {row["id"]: row["text"] for row in public["evidence"]}
        candidate_text = {row["id"]: row["statement"] for row in public["atom_candidates"]}
        targets = {row["evidence_id"]: row for row in scene["target"]["atom_groundings"]}
        fixed = {row["evidence_id"]: row for row in predictions[scene["id"]]["rows"]}
        for evidence in public["evidence"]:
            evidence_id = evidence["id"]
            candidate_id = fixed[evidence_id]["candidate_id"]
            target = targets[evidence_id]
            rows.append({
                "id": "decoder_" + sha256_text(f"26|{scene['id']}|{evidence_id}|{candidate_id}")[:20],
                "schema_version": 26,
                "split": scene["split"],
                "scene_id": scene["id"],
                "episode_id": scene["episode_id"],
                "role": scene["role"],
                "evidence_id": evidence_id,
                "candidate_id": candidate_id,
                "agent_input": {
                    "entities": public["entities"],
                    "action": public["action"],
                    "evidence_text": evidence_text[evidence_id],
                    "candidate_statement": candidate_text[candidate_id],
                    "instruction": "Classify the evidence relative to the candidate fact using the registered letter code.",
                },
                "target": {
                    "truth_label": target["truth_label"],
                    "candidate_assignment_correct": candidate_id == target["candidate_id"],
                },
                "oracle_metadata": {
                    "surface_bank": target["surface_bank"],
                    "semantic_operator": target["semantic_operator"],
                    "predicate_kind": target["predicate_kind"],
                    "relation_orientation": target["relation_orientation"],
                },
            })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/v26-native-truth-decoder.json")
    args = parser.parse_args()
    config_path = (PROJECT_ROOT / args.config).resolve()
    config = json.loads(config_path.read_text())
    output = PROJECT_ROOT / config["outputDir"]
    if output.exists():
        raise RuntimeError("V26 decoder corpus already exists")
    rows = build_rows(config)
    output.mkdir(parents=True)
    for split in ("grounding_fit", "grounding_calibration", "grounding_evaluation"):
        selected = [row for row in rows if row["split"] == split]
        (output / f"{split}.jsonl").write_text(
            "".join(canonical_json(row) + "\n" for row in selected)
        )
    ordered = sorted(rows, key=lambda row: row["id"])
    manifest = {
        "schema_version": 26,
        "experiment": config["experiment"],
        "config": str(config_path.relative_to(PROJECT_ROOT)),
        "config_sha256": file_sha256(config_path),
        "rows": len(rows),
        "split_counts": dict(sorted(Counter(row["split"] for row in rows).items())),
        "role_counts": dict(sorted(Counter(row["role"] for row in rows).items())),
        "target_counts": dict(sorted(Counter(row["target"]["truth_label"] for row in rows).items())),
        "corpus_sha256": sha256_text("".join(canonical_json(row) + "\n" for row in ordered)),
        "source_hashes": {
            key: file_sha256(PROJECT_ROOT / config[key]) for key in (
                "sourceV25Lock", "sourceV25Result", "sourceV25PostAudit",
                "sourceV25Diagnostic", "sourceV24Predictions",
            )
        },
        "new_model_forward_passes": 0,
        "new_linear_fits": 0,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
