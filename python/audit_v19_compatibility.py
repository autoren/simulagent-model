"""Pre-extraction semantic, prompt, firewall, and head-provenance audit for V19."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from transformers import AutoTokenizer

from audit_v18_benchmark import read_records
from build_v19_grounding_views import SUPPORTED_CONCEPTS
from evaluate_v10_frozen import probe, save_pipeline
from evaluate_v15_full_pipeline import nli_pairs_by_base, unique_current_targets
from extract_v10_features_mlx import BASE_SYSTEM_PROMPT, NLI_SYSTEM_PROMPT, base_text, nli_text
from v10_protocol import derive_allowed_values, file_sha256
from v14_protocol import load_records_from_manifest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN_AGENT_KEYS = frozenset({
    "action_dependency_schema", "allowed_values", "assignment", "behavioral_signature",
    "current_value", "executable_schema", "hypothesis_relations", "transition_cases",
})


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def recursive_keys(value: Any) -> set[str]:
    if isinstance(value, dict):
        return set(value).union(*(recursive_keys(entry) for entry in value.values()))
    if isinstance(value, list):
        return set().union(*(recursive_keys(entry) for entry in value)) if value else set()
    return set()


def read_scenes(dataset_dir: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for view in ("supported", "novel_ontology")
        for line in (dataset_dir / f"{view}.jsonl").read_text().splitlines()
        if line
    ]


def expected_items(episodes: Sequence[dict[str, Any]]) -> dict[tuple[str, str], list[dict[str, Any]]]:
    result = {}
    for episode in episodes:
        for grounding in episode["oracle_grounding"]["support"]:
            result[(episode["id"], grounding["trace_id"])] = [
                {
                    "determinant_id": identifier,
                    "allowed_values": ["active" if grounding["assignment"][identifier] else "inactive"],
                }
                for identifier in grounding["assignment"]
            ]
        for query in episode["oracle_grounding"]["queries"]:
            result[(episode["id"], query["query_id"])] = query["allowed_values"]
    return result


def prompt_inventory(scenes: Sequence[dict[str, Any]]) -> tuple[list[str], list[str], dict[str, str]]:
    base_prompts = set()
    nli_prompts = set()
    base_evidence = {}
    for scene in scenes:
        hypotheses = {
            value["determinant_id"]: value["statements"]
            for value in scene["agent_input"]["state_hypotheses"]
        }
        for determinant_index, determinant in enumerate(scene["agent_input"]["transition_determinants"]):
            for evidence_index, unit in enumerate(scene["evidence_units"]):
                base = base_text(scene, determinant_index, evidence_index)
                base_prompts.add(base)
                base_evidence[base] = unit["text"]
                for hypothesis in hypotheses[determinant["id"]]:
                    nli_prompts.add(nli_text(scene, determinant_index, evidence_index, hypothesis))
    return sorted(base_prompts), sorted(nli_prompts), base_evidence


def rendered_prompt_tokens(tokenizer: Any, text: str, system: str) -> tuple[int, list[tuple[int, int]], str]:
    prompt = tokenizer.apply_chat_template(
        [{"role": "system", "content": system}, {"role": "user", "content": text}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    encoded = tokenizer(prompt, add_special_tokens=False, return_offsets_mapping=True)
    return len(encoded["input_ids"]), encoded["offset_mapping"], prompt


def tokenization_audit(
    base_prompts: Sequence[str], nli_prompts: Sequence[str], base_evidence: dict[str, str],
    model: dict[str, Any],
) -> dict[str, Any]:
    tokenizer = AutoTokenizer.from_pretrained(
        model["model"], revision=model["revision"], trust_remote_code=True
    )
    maximum = 0
    empty_spans = 0
    for text in base_prompts:
        count, offsets, prompt = rendered_prompt_tokens(tokenizer, text, BASE_SYSTEM_PROMPT)
        maximum = max(maximum, count)
        evidence = base_evidence[text]
        start = prompt.rfind(evidence)
        end = start + len(evidence)
        if start < 0 or not any(left < end and right > start for left, right in offsets):
            empty_spans += 1
    for text in nli_prompts:
        count, offsets, prompt = rendered_prompt_tokens(tokenizer, text, NLI_SYSTEM_PROMPT)
        maximum = max(maximum, count)
        marker = "Current-state hypothesis: "
        hypothesis = text[text.rfind(marker) + len(marker):]
        start = prompt.rfind(hypothesis)
        end = start + len(hypothesis)
        if start < 0 or not any(left < end and right > start for left, right in offsets):
            empty_spans += 1
    return {
        "tokenizer": model["model"],
        "revision": model["revision"],
        "maximum_prompt_tokens": maximum,
        "empty_target_spans": empty_spans,
        "truncated_prompts": 0 if maximum <= model["maxSequenceLength"] else 1,
    }


def head_provenance_audit(config: dict[str, Any]) -> dict[str, Any]:
    lock_path = PROJECT_ROOT / config["sourceV15Lock"]
    lock = json.loads(lock_path.read_text())
    metadata_path = PROJECT_ROOT / config["sourceV15Features"]
    metadata = json.loads(metadata_path.read_text())
    if metadata["protocol_lock_sha256"] != file_sha256(lock_path):
        raise ValueError("V15 features and lock differ")
    feature_path = PROJECT_ROOT / metadata["feature_artifact"]
    if file_sha256(feature_path) != metadata["feature_artifact_sha256"]:
        raise ValueError("V15 feature artifact hash differs")
    records = load_records_from_manifest(PROJECT_ROOT / lock["source"]["manifest"])
    with np.load(feature_path, allow_pickle=False) as values:
        arrays = {key: values[key] for key in values.files}
    if arrays["record_ids"].tolist() != [record["id"] for record in records]:
        raise ValueError("V15 feature records differ")
    base = arrays["base_span_features"].astype(np.float32)
    match_targets = arrays["unique_base_match_targets"].astype(bool)
    temporal_targets = arrays["unique_base_temporal_targets"].astype(np.int8)
    pair_base = arrays["pair_base_indices"].astype(np.int32)
    current_targets = unique_current_targets(
        pair_base, arrays["current_value_targets"].astype(np.int8), len(base)
    )
    nli_by_base = nli_pairs_by_base(
        pair_base, arrays["pair_nli_indices"].astype(np.int32), len(base)
    )
    nli = arrays["nli_hypothesis_mean_features"].astype(np.float32)
    polarity = nli[nli_by_base[:, 0]] - nli[nli_by_base[:, 1]]
    positive = match_targets
    current = current_targets >= 0
    models = {
        "match": probe(lock["c_value"], lock["seed"]),
        "temporal": probe(lock["c_value"], lock["seed"]),
        "polarity": probe(lock["c_value"], lock["seed"]),
    }
    models["match"].fit(base, match_targets)
    models["temporal"].fit(base[positive], temporal_targets[positive])
    models["polarity"].fit(polarity[current], current_targets[current])
    reproduced: dict[str, np.ndarray] = {}
    for name, model in models.items():
        save_pipeline(name, model, reproduced)
    head_path = PROJECT_ROOT / config["sourceDeploymentHeads"]
    with np.load(head_path, allow_pickle=False) as values:
        frozen = {key: values[key] for key in values.files}
    keys_match = set(reproduced) == set(frozen)
    array_matches = {
        key: bool(np.array_equal(reproduced[key], frozen[key]))
        for key in sorted(set(reproduced) & set(frozen))
    }
    return {
        "passed": keys_match and all(array_matches.values()),
        "head_artifact": config["sourceDeploymentHeads"],
        "head_artifact_sha256": file_sha256(head_path),
        "keys_match": keys_match,
        "array_matches": array_matches,
        "verification_refits": 3,
        "selection_refits": 0,
        "v15_records_read": len(records),
        "v17_head_artifacts_read": 1,
        "v17_records_read": 0,
        "v17_model_results_read": 0,
    }


def audit(
    scenes: Sequence[dict[str, Any]], episodes: Sequence[dict[str, Any]],
    config: dict[str, Any], manifest: dict[str, Any], dataset_dir: Path,
) -> dict[str, Any]:
    errors = []
    expected = expected_items(episodes)
    expected_per_view = len(expected)
    view_counts = Counter(value["view"] for value in scenes)
    if any(view_counts[view] != expected_per_view for view in config["views"]):
        errors.append(f"V19 view counts differ: {dict(view_counts)} expected {expected_per_view}")
    leaked = Counter()
    mismatches = 0
    item_keys: dict[str, set[tuple[str, str]]] = defaultdict(set)
    supported_hypotheses = {
        (value["id"], value["active"], value["inactive"]) for value in SUPPORTED_CONCEPTS
    }
    supported_semantic_mismatches = 0
    scene_grounding_classes: Counter[str] = Counter()
    for scene in scenes:
        leaked.update(recursive_keys(scene["agent_input"]) & FORBIDDEN_AGENT_KEYS)
        key = (scene["episode_id"], scene["source_item_id"])
        item_keys[scene["view"]].add(key)
        if key not in expected:
            mismatches += 1
            continue
        latent_expected = {value["determinant_id"]: value["allowed_values"] for value in expected[key]}
        evidence = scene["evidence_units"]
        if len(evidence) != 4 or len(scene["target"]["determinant_grounding"]) != 4:
            mismatches += 1
        hypotheses = {
            value["determinant_id"]: value["statements"]
            for value in scene["agent_input"]["state_hypotheses"]
        }
        for target in scene["target"]["determinant_grounding"]:
            if target["allowed_values"] != latent_expected[target["latent_determinant_id"]]:
                mismatches += 1
            if derive_allowed_values(target["temporal_status"], target["hypothesis_relations"]) != target["allowed_values"]:
                mismatches += 1
            span = target["evidence_span"]
            if scene["agent_input"]["observation"][span["start"]:span["end"]] != span["text"]:
                mismatches += 1
            if sum(unit == span for unit in evidence) != 1:
                mismatches += 1
            if target["allowed_values"] == ["inactive", "active"]:
                scene_grounding_classes["unresolved"] += 1
            else:
                scene_grounding_classes[target["allowed_values"][0]] += 1
            if scene["view"] == "supported":
                pair = hypotheses[target["determinant_id"]]
                if (target["determinant_id"], pair[0], pair[1]) not in supported_hypotheses:
                    supported_semantic_mismatches += 1
                expected_prefix = (
                    "No current evidence establishes either that "
                    if target["temporal_status"] == "UNKNOWN_CURRENT"
                    else "The present reading shows that "
                )
                if not span["text"].startswith(expected_prefix):
                    supported_semantic_mismatches += 1
        if scene["source"]["v17_records_read"] or scene["source"]["v17_model_results_read"]:
            errors.append(f"{scene['id']} violates the V17 record/result firewall")
    if leaked:
        errors.append(f"V19 agent inputs leak target fields: {dict(leaked)}")
    if mismatches:
        errors.append(f"Found {mismatches} latent-item or grounding mismatches")
    if supported_semantic_mismatches:
        errors.append(f"Found {supported_semantic_mismatches} supported-language mismatches")
    for view in config["views"]:
        if item_keys[view] != set(expected):
            errors.append(f"View {view} does not cover every V18 latent item exactly once")
    if any(value["view_role"] != ("primary" if value["view"] == "supported" else "diagnostic") for value in scenes):
        errors.append("V19 view roles are not isolated")
    artifact_mismatches = 0
    for relative, expected_hash in manifest["artifact_sha256"].items():
        if file_sha256(dataset_dir / relative) != expected_hash:
            artifact_mismatches += 1
    if artifact_mismatches:
        errors.append(f"Found {artifact_mismatches} artifact hash mismatches")

    base_prompts, nli_prompts, base_evidence = prompt_inventory(scenes)
    tokenization = tokenization_audit(base_prompts, nli_prompts, base_evidence, config["model"])
    forward_passes = len(base_prompts) + len(nli_prompts)
    if tokenization["truncated_prompts"] or tokenization["empty_target_spans"]:
        errors.append("V19 prompts fail token-span or truncation checks")
    if forward_passes > config["gates"]["preExtraction"]["maximumNewModelForwardPasses"]:
        errors.append(f"V19 requires {forward_passes} model forwards, above the locked maximum")
    provenance = head_provenance_audit(config)
    if not provenance["passed"]:
        errors.append("Frozen deployment heads do not reproduce from V15 development features")
    if provenance["verification_refits"] > config["gates"]["preExtraction"]["headProvenanceVerificationRefitsPermitted"]:
        errors.append("Head provenance used too many verification refits")

    return {
        "passed": not errors,
        "decision": "authorize_single_v19_feature_extraction" if not errors else "forbid_v19_extraction_revise_interface",
        "errors": errors,
        "scenes": len(scenes),
        "latent_items_per_view": expected_per_view,
        "view_counts": dict(view_counts),
        "agent_input_forbidden_keys": dict(leaked),
        "latent_grounding_mismatches": mismatches,
        "supported_semantic_mismatches": supported_semantic_mismatches,
        "grounding_class_counts": dict(scene_grounding_classes),
        "prompt_inventory": {
            "unique_base_prompts": len(base_prompts),
            "unique_nli_prompts": len(nli_prompts),
            "new_model_forward_passes": forward_passes,
        },
        "tokenization": tokenization,
        "head_provenance": provenance,
        "artifact_hash_mismatches": artifact_mismatches,
        "data_access": {
            "v18_records_read": len(episodes),
            "v17_head_artifacts_read": 1,
            "v17_records_read": 0,
            "v17_model_results_read": 0,
            "adapter_training_runs": 0,
            "new_selection_linear_fits": 0,
            "head_provenance_verification_refits": 3,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/v19-frozen-integration.json")
    parser.add_argument("--dataset", default="data/v19")
    parser.add_argument("--output", default="outputs/v19-frozen-integration/pre-extraction-audit.json")
    args = parser.parse_args()
    config = json.loads((PROJECT_ROOT / args.config).read_text())
    dataset_dir = (PROJECT_ROOT / args.dataset).resolve()
    manifest = json.loads((dataset_dir / "manifest.json").read_text())
    v18_manifest = json.loads((PROJECT_ROOT / config["sourceV18Manifest"]).read_text())
    episodes = read_records((PROJECT_ROOT / config["sourceV18Manifest"]).parent)
    if manifest["source_v18_dataset_sha256"] != v18_manifest["dataset_sha256"]:
        raise ValueError("V19 and V18 manifests do not share a dataset hash")
    result = audit(read_scenes(dataset_dir), episodes, config, manifest, dataset_dir)
    output = (PROJECT_ROOT / args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
