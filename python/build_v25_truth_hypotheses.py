"""Build the immutable V25 truth-hypothesis corpus from fixed V24 assignments."""

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
    v24_lock = json.loads((PROJECT_ROOT / config["sourceV24Lock"]).read_text())
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
        target = {row["evidence_id"]: row for row in scene["target"]["atom_groundings"]}
        fixed = {row["evidence_id"]: row for row in predictions[scene["id"]]["rows"]}
        for evidence in public["evidence"]:
            evidence_id = evidence["id"]
            target_row = target[evidence_id]
            candidate_sources = {fixed[evidence_id]["candidate_id"]: ["v24_fixed_assignment"]}
            if scene["split"] == config["head"]["fitSplit"]:
                candidate_sources.setdefault(target_row["candidate_id"], []).append("fit_gold_candidate")
            for candidate_id, sources in sorted(candidate_sources.items()):
                for hypothesis in config["assessmentHypotheses"]:
                    row_id = "truthpair_" + sha256_text(
                        f"25|{scene['id']}|{evidence_id}|{candidate_id}|{hypothesis['id']}"
                    )[:20]
                    rows.append({
                        "id": row_id,
                        "schema_version": 25,
                        "split": scene["split"],
                        "scene_id": scene["id"],
                        "episode_id": scene["episode_id"],
                        "role": scene["role"],
                        "evidence_id": evidence_id,
                        "candidate_id": candidate_id,
                        "assessment_id": hypothesis["id"],
                        "selection_sources": sources,
                        "agent_input": {
                            "entities": public["entities"],
                            "action": public["action"],
                            "evidence_text": evidence_text[evidence_id],
                            "candidate_statement": candidate_text[candidate_id],
                            "assessment_statement": hypothesis["statement"],
                            "instruction": "Represent whether the assessment hypothesis fits the evidence and candidate fact.",
                        },
                        "target": {
                            "compatible": hypothesis["truthLabel"] == target_row["truth_label"],
                            "truth_label": target_row["truth_label"],
                            "assessment_truth_label": hypothesis["truthLabel"],
                            "candidate_assignment_correct": candidate_id == target_row["candidate_id"],
                            "use_for_fit": (
                                scene["split"] == config["head"]["fitSplit"]
                                and candidate_id == target_row["candidate_id"]
                            ),
                        },
                        "oracle_metadata": {
                            "surface_bank": target_row["surface_bank"],
                            "semantic_operator": target_row["semantic_operator"],
                            "predicate_kind": target_row["predicate_kind"],
                            "relation_orientation": target_row["relation_orientation"],
                        },
                    })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/v25-truth-hypotheses.json")
    args = parser.parse_args()
    config_path = (PROJECT_ROOT / args.config).resolve()
    config = json.loads(config_path.read_text())
    output = PROJECT_ROOT / config["outputDir"]
    if output.exists():
        raise RuntimeError("V25 truth-hypothesis corpus already exists")
    rows = build_rows(config)
    output.mkdir(parents=True)
    for split in ("grounding_fit", "grounding_calibration", "grounding_evaluation"):
        selected = [row for row in rows if row["split"] == split]
        (output / f"{split}.jsonl").write_text(
            "".join(canonical_json(row) + "\n" for row in selected)
        )
    ordered = sorted(rows, key=lambda row: row["id"])
    manifest = {
        "schema_version": 25,
        "experiment": config["experiment"],
        "config": str(config_path.relative_to(PROJECT_ROOT)),
        "config_sha256": file_sha256(config_path),
        "rows": len(rows),
        "base_pairs": len({
            (row["scene_id"], row["evidence_id"], row["candidate_id"]) for row in rows
        }),
        "fixed_assignment_evidence_groups": len({
            (row["scene_id"], row["evidence_id"]) for row in rows
            if "v24_fixed_assignment" in row["selection_sources"]
        }),
        "fit_rows": sum(row["target"]["use_for_fit"] for row in rows),
        "fit_positive_rows": sum(
            row["target"]["use_for_fit"] and row["target"]["compatible"] for row in rows
        ),
        "split_counts": dict(sorted(Counter(row["split"] for row in rows).items())),
        "assessment_counts": dict(sorted(Counter(row["assessment_id"] for row in rows).items())),
        "corpus_sha256": sha256_text("".join(canonical_json(row) + "\n" for row in ordered)),
        "source_hashes": {
            key: file_sha256(PROJECT_ROOT / config[key]) for key in (
                "sourceV24Lock", "sourceV24Result", "sourceV24PostAudit",
                "sourceV24Diagnostic", "sourceV24FeatureMetadata", "sourceV24Heads",
                "sourceV24Predictions",
            )
        },
        "new_model_forward_passes": 0,
        "new_linear_fits": 0,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
