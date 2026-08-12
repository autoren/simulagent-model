"""Build V15-compatible language views over the unchanged V18 latent episodes."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

from audit_v18_benchmark import read_records
from generate_v18_schema_benchmark import LEXICONS


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CANDIDATE_ACTION = "use the tuning fork at the beacon console"
SUPPORTED_CONCEPTS = (
    {
        "id": "generator_stable", "label": "the generator rhythm is stable",
        "active": "the generator rhythm is even", "inactive": "the generator output surges unevenly",
    },
    {
        "id": "mirror_seated", "label": "the mirror shard is seated",
        "active": "the mirror shard sits flush in its socket", "inactive": "the mirror socket is empty",
    },
    {
        "id": "fork_calibrated", "label": "the carried tuning fork is calibrated",
        "active": "the fork tone matches the reference pitch", "inactive": "the fork tone falls away from the reference pitch",
    },
    {
        "id": "hatch_unlocked", "label": "the observatory hatch is unlocked",
        "active": "the observatory hatch stands unlatched", "inactive": "the observatory hatch remains latched",
    },
)


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def concepts_for(record: dict[str, Any], view: str) -> Sequence[dict[str, str]]:
    if view == "supported":
        return SUPPORTED_CONCEPTS
    if view == "novel_ontology":
        return LEXICONS[record["lexicon_family"]]
    raise ValueError(f"Unknown V19 view {view}")


def render_unit(concept: dict[str, str], allowed_values: list[str]) -> tuple[str, str, str | None, list[str]]:
    if allowed_values == ["active"]:
        return (
            f"The present reading shows that {concept['active']}.",
            "CURRENT", "active", ["ENTAILED", "CONTRADICTED"],
        )
    if allowed_values == ["inactive"]:
        return (
            f"The present reading shows that {concept['inactive']}.",
            "CURRENT", "inactive", ["CONTRADICTED", "ENTAILED"],
        )
    if allowed_values == ["inactive", "active"]:
        return (
            f"No current evidence establishes either that {concept['active']} or that {concept['inactive']}.",
            "UNKNOWN_CURRENT", None, ["UNKNOWN", "UNKNOWN"],
        )
    raise ValueError(f"Unsupported V19 allowed values {allowed_values}")


def build_scene(
    episode: dict[str, Any],
    view: str,
    item_kind: str,
    item_id: str,
    latent_grounding: list[dict[str, Any]],
) -> dict[str, Any]:
    concepts = concepts_for(episode, view)
    latent_ids = [value["id"] for value in episode["agent_input"]["determinant_ontology"]]
    allowed_by_id = {value["determinant_id"]: value["allowed_values"] for value in latent_grounding}
    rendered = []
    for position, (latent_id, concept) in enumerate(zip(latent_ids, concepts, strict=True)):
        text, temporal, current, relations = render_unit(concept, allowed_by_id[latent_id])
        rendered.append({
            "position": position,
            "latent_id": latent_id,
            "concept": concept,
            "text": text,
            "temporal": temporal,
            "current": current,
            "relations": relations,
            "allowed_values": allowed_by_id[latent_id],
            "order": sha256_text(f"{episode['id']}|{view}|{item_id}|{concept['id']}")[:16],
        })
    rendered.sort(key=lambda value: value["order"])
    observation = ""
    evidence_units = []
    span_by_position = {}
    for value in rendered:
        if observation:
            observation += "\n"
        start = len(observation)
        observation += value["text"]
        evidence = {"start": start, "end": len(observation), "text": value["text"]}
        evidence_units.append(evidence)
        span_by_position[value["position"]] = evidence
    determinant_grounding = []
    for position, (latent_id, concept) in enumerate(zip(latent_ids, concepts, strict=True)):
        value = next(entry for entry in rendered if entry["position"] == position)
        determinant_grounding.append({
            "determinant_id": concept["id"],
            "latent_determinant_id": latent_id,
            "temporal_status": value["temporal"],
            "current_value": value["current"],
            "hypothesis_relations": value["relations"],
            "allowed_values": value["allowed_values"],
            "evidence_span": span_by_position[position],
        })
    scene_key = sha256_text(f"{episode['id']}|{view}|{item_kind}|{item_id}")[:24]
    return {
        "id": f"v19:{scene_key}",
        "schema_version": 19,
        "split": episode["split"],
        "generalization_axis": episode["generalization_axis"],
        "episode_id": episode["id"],
        "view": view,
        "view_role": "primary" if view == "supported" else "diagnostic",
        "item_kind": item_kind,
        "source_item_id": item_id,
        "agent_input": {
            "task": "ground_current_state_polarity",
            "candidate_action": CANDIDATE_ACTION,
            "transition_determinants": [
                {"id": value["id"], "label": value["label"]} for value in concepts
            ],
            "state_hypotheses": [
                {"determinant_id": value["id"], "statements": [value["active"], value["inactive"]]}
                for value in concepts
            ],
            "observation": observation,
            "output_instruction": (
                "Match each determinant to one evidence unit, classify its temporal status, "
                "and compare reliable current evidence with both supplied state hypotheses."
            ),
        },
        "evidence_units": evidence_units,
        "target": {"determinant_grounding": determinant_grounding},
        "source": {
            "v18_episode_id": episode["id"],
            "v18_dataset_sha256": None,
            "v17_records_read": 0,
            "v17_model_results_read": 0,
        },
    }


def build(records: Sequence[dict[str, Any]], dataset_sha256: str) -> list[dict[str, Any]]:
    scenes = []
    for episode in records:
        observed = {
            value["trace_id"]: value["observed_transition_code"]
            for value in episode["agent_input"]["support_traces"]
        }
        for view in ("supported", "novel_ontology"):
            for grounding in episode["oracle_grounding"]["support"]:
                allowed = [
                    {
                        "determinant_id": identifier,
                        "allowed_values": ["active" if grounding["assignment"][identifier] else "inactive"],
                    }
                    for identifier in grounding["assignment"]
                ]
                scene = build_scene(episode, view, "support", grounding["trace_id"], allowed)
                scene["observed_transition_code"] = observed[grounding["trace_id"]]
                scene["source"]["v18_dataset_sha256"] = dataset_sha256
                scenes.append(scene)
            for query in episode["oracle_grounding"]["queries"]:
                scene = build_scene(episode, view, "query", query["query_id"], query["allowed_values"])
                scene["source"]["v18_dataset_sha256"] = dataset_sha256
                scenes.append(scene)
    return scenes


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/v19-frozen-integration.json")
    args = parser.parse_args()
    config_path = (PROJECT_ROOT / args.config).resolve()
    config_text = config_path.read_text()
    config = json.loads(config_text)
    v18_manifest_path = (PROJECT_ROOT / config["sourceV18Manifest"]).resolve()
    v18_manifest = json.loads(v18_manifest_path.read_text())
    v18_records = read_records(v18_manifest_path.parent)
    scenes = build(v18_records, v18_manifest["dataset_sha256"])
    output_dir = (PROJECT_ROOT / config["outputDir"]).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    artifact_hashes = {}
    dataset_parts = []
    for view in ("supported", "novel_ontology"):
        selected = [value for value in scenes if value["view"] == view]
        content = "".join(canonical_json(value) + "\n" for value in selected)
        relative = f"{view}.jsonl"
        (output_dir / relative).write_text(content)
        artifact_hashes[relative] = sha256_text(content)
        dataset_parts.append(f"{relative}\n{content}")
    manifest = {
        "schema_version": 19,
        "experiment": config["experiment"],
        "config": config,
        "config_sha256": sha256_text(config_text),
        "source_v18_manifest": config["sourceV18Manifest"],
        "source_v18_manifest_sha256": sha256_text(v18_manifest_path.read_text()),
        "source_v18_dataset_sha256": v18_manifest["dataset_sha256"],
        "scenes": len(scenes),
        "view_counts": dict(sorted({view: sum(value["view"] == view for value in scenes) for view in config["views"]}.items())),
        "artifact_sha256": artifact_hashes,
        "implementation_sha256": {
            "python/build_v19_grounding_views.py": sha256_text(Path(__file__).read_text()),
        },
        "dataset_sha256": sha256_text("".join(dataset_parts)),
        "data_access": {
            "v18_records_read": len(v18_records),
            "v17_records_read": 0,
            "v17_model_results_read": 0,
            "adapter_training_runs": 0,
        },
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
