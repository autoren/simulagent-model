"""Pre-decoder structural, search-space, and firewall audit for V27."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

from audit_v22r2_grounding import read_jsonl_directory
from audit_v24_cross_encoder import read_pairs
from build_v27_support_edges import build_rows
from v10_protocol import file_sha256
from v22_relational import canonical_json
from v22r2_grounding import PROJECT_ROOT
from v24_cross_encoder import sha256_text
from v26_native_decoder import decoder_prompt


FORBIDDEN_PUBLIC_KEYS = {
    "allowed_values", "atom", "candidate_assignment_correct", "compatible",
    "epistemic_state", "possible_transition_codes", "program", "query_axis",
    "same_atom", "truth_label",
}


def recursive_keys(value: Any) -> Iterable[str]:
    if isinstance(value, dict):
        for key, child in value.items():
            yield key
            yield from recursive_keys(child)
    elif isinstance(value, list):
        for child in value:
            yield from recursive_keys(child)


def read_rows(root: Path) -> list[dict[str, Any]]:
    rows = []
    for split in ("grounding_fit", "grounding_calibration", "grounding_evaluation"):
        rows.extend(
            json.loads(line) for line in (root / f"{split}.jsonl").read_text().splitlines()
            if line.strip()
        )
    return rows


def count_sparse_matchings(scene, proposals):
    evidence = [row["id"] for row in scene["agent_input"]["evidence"]]
    used = set()
    count = 0

    def visit(index):
        nonlocal count
        if index == len(evidence):
            count += 1
            return
        for candidate in sorted(proposals[(scene["id"], evidence[index])]):
            if candidate not in used:
                used.add(candidate)
                visit(index + 1)
                used.remove(candidate)
    visit(0)
    return count


def audit(rows, config, manifest, scenes, support_pairs, v26_scores):
    errors = []
    scene_lookup = {row["id"]: row for row in scenes}
    support_scenes = [row for row in scenes if row["role"] == "support"]
    all_edges = {
        (row["scene_id"], row["evidence_id"], row["candidate_id"]): row
        for row in support_pairs
    }
    reused = {
        (row["scene_id"], row["evidence_id"], row["candidate_id"])
        for row in v26_scores if row["role"] == "support"
    }
    new = {(row["scene_id"], row["evidence_id"], row["candidate_id"]) for row in rows}
    if reused & new or reused | new != set(all_edges):
        errors.append("V27 reused/new edge partition does not equal all V24 support proposals")
    leaks = Counter()
    exact_inputs = 0
    for row in rows:
        pair = all_edges[(row["scene_id"], row["evidence_id"], row["candidate_id"])]
        exact_inputs += row["agent_input"] == pair["agent_input"]
        for key in set(recursive_keys(row["agent_input"])) & FORBIDDEN_PUBLIC_KEYS:
            leaks[key] += 1
        if row["role"] != "support" or scene_lookup[row["scene_id"]]["role"] != "support":
            errors.append(f"Non-support row appears in V27: {row['id']}")
    if leaks:
        errors.append(f"Target/oracle fields leaked into V27 prompts: {dict(leaks)}")
    if exact_inputs != len(rows):
        errors.append("V27 prompt inputs differ from their V24 source pairs")

    proposals = defaultdict(set)
    positive = set()
    for pair in support_pairs:
        proposals[(pair["scene_id"], pair["evidence_id"])].add(pair["candidate_id"])
        if pair["target"]["same_atom"]:
            positive.add((pair["scene_id"], pair["evidence_id"]))
    matching_counts = {scene["id"]: count_sparse_matchings(scene, proposals) for scene in support_scenes}
    maximum = max(matching_counts.values())
    if maximum > config["jointMap"]["maximumAssignmentsPerScene"]:
        errors.append("V27 exhaustive sparse assignment limit is too small")
    coverage = {}
    exact_scene = {}
    exact_episode = {}
    for split in ("grounding_fit", "grounding_calibration", "grounding_evaluation"):
        selected = [scene for scene in support_scenes if scene["split"] == split]
        evidence_flags = [
            (scene["id"], evidence["id"]) in positive
            for scene in selected for evidence in scene["agent_input"]["evidence"]
        ]
        scene_flags = {
            scene["id"]: all(
                (scene["id"], evidence["id"]) in positive
                for evidence in scene["agent_input"]["evidence"]
            ) for scene in selected
        }
        by_episode = defaultdict(list)
        for scene in selected:
            by_episode[scene["episode_id"]].append(scene_flags[scene["id"]])
        coverage[split] = sum(evidence_flags) / len(evidence_flags)
        exact_scene[split] = sum(scene_flags.values()) / len(scene_flags)
        exact_episode[split] = sum(all(values) for values in by_episode.values()) / len(by_episode)
    gates = config["gates"]["preEvaluation"]
    if coverage["grounding_evaluation"] < gates["minimumEvaluationSupportGoldEdgeCoverage"]:
        errors.append("Evaluation support gold-edge coverage misses the V27 gate")
    if exact_scene["grounding_evaluation"] < gates["minimumEvaluationSupportExactProposalSceneCoverage"]:
        errors.append("Evaluation exact support proposal-scene coverage misses the V27 gate")
    if len(rows) > gates["maximumNewModelForwardPasses"]:
        errors.append("V27 new decoder corpus exceeds its forward budget")
    prompts = defaultdict(set)
    for row in rows:
        prompts[row["split"]].add(decoder_prompt(row))
    overlap = prompts["grounding_fit"] & prompts["grounding_evaluation"]
    if overlap:
        errors.append("Exact V27 fit/evaluation decoder prompts overlap")

    ordered = sorted(rows, key=lambda row: row["id"])
    corpus_hash = sha256_text("".join(canonical_json(row) + "\n" for row in ordered))
    if manifest["new_decoder_rows"] != len(rows) or manifest["corpus_sha256"] != corpus_hash:
        errors.append("V27 corpus differs from its manifest")
    if manifest["config_sha256"] != file_sha256(PROJECT_ROOT / manifest["config"]):
        errors.append("V27 config differs from its manifest")
    for key, expected in manifest["source_hashes"].items():
        if file_sha256(PROJECT_ROOT / config[key]) != expected:
            errors.append(f"V27 source changed after corpus construction: {key}")
    v26_audit = json.loads((PROJECT_ROOT / config["sourceV26PostAudit"]).read_text())
    v26_result = json.loads((PROJECT_ROOT / config["sourceV26Result"]).read_text())
    residual = json.loads((PROJECT_ROOT / config["sourceV26Residual"]).read_text())
    if not v26_audit["passed"] or v26_audit["decision"] != "accept_v26_exposed_development_result":
        errors.append("V26 integrity status does not authorize V27")
    if v26_result["decision"] != "repair_exact_graph_or_symbolic_composition_no_lora":
        errors.append("V26 registered decision does not authorize V27")
    if not residual["localization"]["support_stage_is_primary_remaining_bottleneck"]:
        errors.append("V26 residual does not localize V27 to support")
    return {
        "schema_version": 27,
        "experiment": "v27_pre_decoder_support_map_audit",
        "passed": not errors,
        "decision": "authorize_v27_protocol_lock" if not errors else "repair_v27_before_model_access",
        "errors": errors,
        "population": {
            "support_scenes": len(support_scenes),
            "support_evidence_groups": len(proposals),
            "all_support_proposal_edges": len(all_edges),
            "reused_v26_edges": len(reused),
            "new_decoder_rows": len(rows),
        },
        "proposal": {
            "gold_edge_coverage_by_split": coverage,
            "exact_scene_coverage_by_split": exact_scene,
            "exact_episode_coverage_by_split": exact_episode,
            "maximum_sparse_perfect_matchings": maximum,
            "matching_count_distribution": {
                str(key): value for key, value in sorted(Counter(matching_counts.values()).items())
            },
        },
        "firewall": {
            "forbidden_agent_input_keys": dict(sorted(leaks.items())),
            "exact_source_agent_inputs": exact_inputs,
            "exact_fit_evaluation_prompt_overlap": len(overlap),
            "new_model_forward_passes_before_lock": 0,
            "new_linear_fits_before_lock": 0,
            "fresh_benchmark_records_created": 0,
        },
        "budget": {
            "planned_new_model_forwards": len(rows),
            "registered_new_model_forwards": config["limits"]["newModelForwardPasses"],
        },
        "integrity": {"corpus_sha256": corpus_hash},
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/v27-support-map.json")
    parser.add_argument("--output", default="outputs/v27-support-map/pre-decoder-audit.json")
    args = parser.parse_args()
    config = json.loads((PROJECT_ROOT / args.config).read_text())
    root = PROJECT_ROOT / config["outputDir"]
    rows = read_rows(root)
    manifest = json.loads((root / "manifest.json").read_text())
    v26_lock = json.loads((PROJECT_ROOT / config["sourceV26Lock"]).read_text())
    v25_lock = json.loads((PROJECT_ROOT / v26_lock["source"]["v25_lock"]).read_text())
    v24_lock = json.loads((PROJECT_ROOT / v25_lock["source"]["v24_lock"]).read_text())
    original_lock = json.loads((PROJECT_ROOT / v24_lock["source"]["v22r2_lock"]).read_text())
    scenes = read_jsonl_directory(PROJECT_ROOT / original_lock["source"]["dataset"] / "scenes")
    support_pairs = [
        row for row in read_pairs(PROJECT_ROOT / config["sourceV24ProposalCorpus"])
        if row["role"] == "support"
    ]
    v26_scores = [
        json.loads(line) for line in (PROJECT_ROOT / config["sourceV26Scores"]).read_text().splitlines()
        if line.strip()
    ]
    result = audit(rows, config, manifest, scenes, support_pairs, v26_scores)
    output = (PROJECT_ROOT / args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
