"""Build the immutable V24 top-3-plus-hard-assignment pair corpus."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from audit_v22r2_grounding import read_jsonl_directory
from evaluate_v22r2_relational_grounding import load_npz
from v10_protocol import file_sha256
from v22_relational import canonical_json
from v22r2_grounding import PROJECT_ROOT
from v24_cross_encoder import old_match_scores, proposal_candidate_ids, sha256_text


def feature_maps(arrays: dict[str, np.ndarray]):
    return (
        {str(key): arrays["candidate_features"][i] for i, key in enumerate(arrays["candidate_ids"].tolist())},
        {str(key): arrays["evidence_features"][i] for i, key in enumerate(arrays["evidence_ids"].tolist())},
    )


def build_pairs(config: dict[str, Any]) -> list[dict[str, Any]]:
    original_lock = json.loads((PROJECT_ROOT / config["sourceV22r2Lock"]).read_text())
    metadata = json.loads((PROJECT_ROOT / config["sourceFeatures"]).read_text())
    arrays = load_npz(PROJECT_ROOT / metadata["feature_artifact"])
    heads = load_npz(PROJECT_ROOT / config["sourceHeads"])
    candidate_features, evidence_features = feature_maps(arrays)
    scenes = read_jsonl_directory(PROJECT_ROOT / original_lock["source"]["dataset"] / "scenes")
    predictions = {
        row["scene_id"]: row for row in [
            json.loads(line) for line in (PROJECT_ROOT / config["sourcePredictions"]).read_text().splitlines()
            if line.strip()
        ]
    }
    atom_coef = heads["atom_coef"][0]
    atom_intercept = float(heads["atom_intercept"][0])
    top_k = config["proposal"]["perEvidenceRawTopK"]
    pairs = []
    for scene in scenes:
        public = scene["agent_input"]
        candidate_ids = [row["id"] for row in public["atom_candidates"]]
        candidate_statements = {row["id"]: row["statement"] for row in public["atom_candidates"]}
        candidate_x = np.stack([candidate_features[value] for value in candidate_ids])
        prediction = {row["evidence_id"]: row for row in predictions[scene["id"]]["rows"]}
        target_by_evidence = {
            row["evidence_id"]: row for row in scene["target"]["atom_groundings"]
        }
        evidence_texts = {row["id"]: row["text"] for row in public["evidence"]}
        for evidence_id in [row["id"] for row in public["evidence"]]:
            scores = old_match_scores(
                candidate_x, evidence_features[evidence_id], atom_coef, atom_intercept
            )
            proposals = proposal_candidate_ids(
                candidate_ids, scores, prediction[evidence_id]["candidate_id"], top_k
            )
            target = target_by_evidence[evidence_id]
            for proposal in proposals:
                candidate_id = proposal["candidate_id"]
                same = candidate_id == target["candidate_id"]
                pair_id = "pair_" + sha256_text(
                    f"24|{scene['id']}|{evidence_id}|{candidate_id}"
                )[:20]
                pairs.append({
                    "id": pair_id,
                    "schema_version": 24,
                    "split": scene["split"],
                    "scene_id": scene["id"],
                    "episode_id": scene["episode_id"],
                    "role": scene["role"],
                    "evidence_id": evidence_id,
                    "candidate_id": candidate_id,
                    "agent_input": {
                        "entities": public["entities"],
                        "action": public["action"],
                        "evidence_text": evidence_texts[evidence_id],
                        "candidate_statement": candidate_statements[candidate_id],
                        "instruction": "Compare the evidence statement with the exact candidate fact.",
                    },
                    "proposal": proposal,
                    "target": {
                        "same_atom": same,
                        "pair_class": target["truth_label"] if same else "other",
                        "truth_label": target["truth_label"] if same else None,
                    },
                    "oracle_metadata": {
                        "predicate_kind": target["predicate_kind"],
                        "relation_orientation": target["relation_orientation"],
                        "semantic_operator": target["semantic_operator"],
                        "surface_bank": target["surface_bank"],
                        "query_axis": scene["oracle_metadata"].get("query_axis"),
                    },
                })
    return pairs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/v24-cross-encoder.json")
    args = parser.parse_args()
    config_path = (PROJECT_ROOT / args.config).resolve()
    config = json.loads(config_path.read_text())
    output = PROJECT_ROOT / config["outputDir"]
    if output.exists():
        raise RuntimeError("V24 proposal corpus already exists")
    pairs = build_pairs(config)
    output.mkdir(parents=True)
    for split in ("grounding_fit", "grounding_calibration", "grounding_evaluation"):
        rows = [row for row in pairs if row["split"] == split]
        (output / f"{split}.jsonl").write_text("".join(canonical_json(row) + "\n" for row in rows))
    ordered = sorted(pairs, key=lambda row: row["id"])
    manifest = {
        "schema_version": 24,
        "experiment": config["experiment"],
        "config": str(config_path.relative_to(PROJECT_ROOT)),
        "config_sha256": file_sha256(config_path),
        "pairs": len(pairs),
        "split_counts": dict(sorted(Counter(row["split"] for row in pairs).items())),
        "role_counts": dict(sorted(Counter(row["role"] for row in pairs).items())),
        "class_counts": dict(sorted(Counter(row["target"]["pair_class"] for row in pairs).items())),
        "corpus_sha256": sha256_text("".join(canonical_json(row) + "\n" for row in ordered)),
        "source_hashes": {
            key: file_sha256(PROJECT_ROOT / config[key])
            for key in (
                "sourceV22r2Lock", "sourceV22r2aLock", "sourceV22r2aResult",
                "sourceV22r2aPostAudit", "sourceV22r2aDiagnostic", "sourceV23Result",
                "sourceV23PostAudit", "sourceFeatures", "sourceHeads", "sourcePredictions",
            )
        },
        "new_model_forward_passes": 0,
        "new_linear_fits": 0,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
