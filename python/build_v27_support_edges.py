"""Materialize V27's unscored support proposal edges."""

from __future__ import annotations

import argparse
import json
from collections import Counter

from audit_v24_cross_encoder import read_pairs
from v10_protocol import file_sha256
from v22_relational import canonical_json
from v22r2_grounding import PROJECT_ROOT
from v24_cross_encoder import sha256_text


def build_rows(config):
    pairs = read_pairs(PROJECT_ROOT / config["sourceV24ProposalCorpus"])
    support = [row for row in pairs if row["role"] == "support"]
    scores = [
        json.loads(line) for line in (PROJECT_ROOT / config["sourceV26Scores"]).read_text().splitlines()
        if line.strip()
    ]
    scored = {(row["scene_id"], row["evidence_id"], row["candidate_id"]) for row in scores}
    rows = []
    for pair in support:
        key = (pair["scene_id"], pair["evidence_id"], pair["candidate_id"])
        if key in scored:
            continue
        rows.append({
            "id": "supportedge_" + sha256_text(
                f"27|{pair['scene_id']}|{pair['evidence_id']}|{pair['candidate_id']}"
            )[:20],
            "schema_version": 27,
            "split": pair["split"],
            "scene_id": pair["scene_id"],
            "episode_id": pair["episode_id"],
            "role": "support",
            "evidence_id": pair["evidence_id"],
            "candidate_id": pair["candidate_id"],
            "source_pair_id": pair["id"],
            "agent_input": pair["agent_input"],
            "target": pair["target"],
            "oracle_metadata": pair["oracle_metadata"],
        })
    return rows, support, scored


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/v27-support-map.json")
    args = parser.parse_args()
    config_path = (PROJECT_ROOT / args.config).resolve()
    config = json.loads(config_path.read_text())
    output = PROJECT_ROOT / config["outputDir"]
    if output.exists():
        raise RuntimeError("V27 support-edge corpus already exists")
    rows, support, scored = build_rows(config)
    output.mkdir(parents=True)
    for split in ("grounding_fit", "grounding_calibration", "grounding_evaluation"):
        selected = [row for row in rows if row["split"] == split]
        (output / f"{split}.jsonl").write_text(
            "".join(canonical_json(row) + "\n" for row in selected)
        )
    ordered = sorted(rows, key=lambda row: row["id"])
    manifest = {
        "schema_version": 27,
        "experiment": config["experiment"],
        "config": str(config_path.relative_to(PROJECT_ROOT)),
        "config_sha256": file_sha256(config_path),
        "all_support_proposal_edges": len(support),
        "reused_v26_edges": len({
            (row["scene_id"], row["evidence_id"], row["candidate_id"])
            for row in support if (row["scene_id"], row["evidence_id"], row["candidate_id"]) in scored
        }),
        "new_decoder_rows": len(rows),
        "split_counts": dict(sorted(Counter(row["split"] for row in rows).items())),
        "corpus_sha256": sha256_text("".join(canonical_json(row) + "\n" for row in ordered)),
        "source_hashes": {
            key: file_sha256(PROJECT_ROOT / config[key]) for key in (
                "sourceV26Lock", "sourceV26Result", "sourceV26PostAudit", "sourceV26Residual",
                "sourceV26Scores", "sourceV26Predictions", "sourceV24Features", "sourceV24Heads",
            )
        },
        "source_v24_proposal_manifest_sha256": file_sha256(
            PROJECT_ROOT / config["sourceV24ProposalCorpus"] / "manifest.json"
        ),
        "new_model_forward_passes": 0,
        "new_linear_fits": 0,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
