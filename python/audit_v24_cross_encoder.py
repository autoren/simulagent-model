"""Pre-extraction proposal, source-integrity, and firewall audit for V24."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

from audit_v22r2_grounding import read_jsonl_directory
from v10_protocol import file_sha256
from v22_relational import canonical_json
from v22r2_grounding import PROJECT_ROOT
from v24_cross_encoder import cross_prompt_text, sha256_text


FORBIDDEN_PUBLIC_KEYS = {
    "allowed_values", "atom", "atom_groundings", "candidate_id", "epistemic_state",
    "pair_class", "possible_transition_codes", "program", "program_key", "query_axis",
    "reference_complete_world", "same_atom", "semantic_operator", "truth_label",
}


def recursive_keys(value: Any) -> Iterable[str]:
    if isinstance(value, dict):
        for key, child in value.items():
            yield key
            yield from recursive_keys(child)
    elif isinstance(value, list):
        for child in value:
            yield from recursive_keys(child)


def read_pairs(root: Path) -> list[dict[str, Any]]:
    rows = []
    for split in ("grounding_fit", "grounding_calibration", "grounding_evaluation"):
        path = root / f"{split}.jsonl"
        rows.extend(json.loads(line) for line in path.read_text().splitlines() if line.strip())
    return rows


def source_maps(
    scenes: Sequence[dict[str, Any]], prediction_rows: Sequence[dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    return (
        {row["id"]: row for row in scenes},
        {row["scene_id"]: row for row in prediction_rows},
    )


def audit(
    pairs: Sequence[dict[str, Any]], config: dict[str, Any], manifest: dict[str, Any],
    scenes: Sequence[dict[str, Any]], prediction_rows: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    errors: list[str] = []
    scene_by_id, prediction_by_scene = source_maps(scenes, prediction_rows)
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    prompts: dict[str, set[str]] = defaultdict(set)
    leaked_keys = Counter()
    exact_agent_target_matches = 0
    for pair in pairs:
        groups[(pair["scene_id"], pair["evidence_id"])].append(pair)
        prompts[pair["split"]].add(cross_prompt_text(pair))
        for key in set(recursive_keys(pair["agent_input"])) & FORBIDDEN_PUBLIC_KEYS:
            leaked_keys[key] += 1
        scene = scene_by_id.get(pair["scene_id"])
        if scene is None:
            errors.append(f"Pair {pair['id']} references an unknown scene")
            continue
        public = scene["agent_input"]
        evidence = {row["id"]: row["text"] for row in public["evidence"]}
        candidates = {row["id"]: row["statement"] for row in public["atom_candidates"]}
        expected_agent = {
            "entities": public["entities"],
            "action": public["action"],
            "evidence_text": evidence[pair["evidence_id"]],
            "candidate_statement": candidates[pair["candidate_id"]],
            "instruction": "Compare the evidence statement with the exact candidate fact.",
        }
        exact_agent_target_matches += pair["agent_input"] == expected_agent
        if pair["split"] != scene["split"] or pair["role"] != scene["role"]:
            errors.append(f"Pair {pair['id']} split or role differs from its source scene")
    if leaked_keys:
        errors.append(f"Forbidden target/oracle keys occur in agent inputs: {dict(leaked_keys)}")
    if exact_agent_target_matches != len(pairs):
        errors.append("One or more pair agent inputs differ from their registered source fields")

    proposal_sizes = Counter()
    coverage: dict[tuple[str, str], list[bool]] = defaultdict(list)
    hard_edges_present = 0
    perfect_matching_scenes = 0
    for (scene_id, evidence_id), rows in groups.items():
        proposal_sizes[len(rows)] += 1
        candidate_ids = [row["candidate_id"] for row in rows]
        if len(set(candidate_ids)) != len(candidate_ids):
            errors.append(f"Duplicate proposed candidate for {scene_id}/{evidence_id}")
        minimum = config["proposal"]["minimumEdgesPerEvidence"]
        maximum = config["proposal"]["maximumEdgesPerEvidence"]
        if not minimum <= len(rows) <= maximum:
            errors.append(f"Proposal count outside registration for {scene_id}/{evidence_id}")
        source_prediction = {
            row["evidence_id"]: row
            for row in prediction_by_scene[scene_id]["rows"]
        }[evidence_id]
        hard_id = source_prediction["candidate_id"]
        hard_rows = [
            row for row in rows
            if "global_hard_assignment" in row["proposal"]["proposal_sources"]
        ]
        if len(hard_rows) != 1 or hard_rows[0]["candidate_id"] != hard_id:
            errors.append(f"Registered hard-assignment edge is not preserved for {scene_id}/{evidence_id}")
        else:
            hard_edges_present += 1
        top_rows = [
            row for row in rows if "raw_top_k" in row["proposal"]["proposal_sources"]
        ]
        if len(top_rows) != config["proposal"]["perEvidenceRawTopK"]:
            errors.append(f"Raw top-k cardinality differs for {scene_id}/{evidence_id}")
        if sorted(row["proposal"]["raw_rank"] for row in top_rows) != list(
            range(1, config["proposal"]["perEvidenceRawTopK"] + 1)
        ):
            errors.append(f"Raw proposal ranks differ for {scene_id}/{evidence_id}")
        positive = [row for row in rows if row["target"]["same_atom"]]
        if len(positive) > 1:
            errors.append(f"Multiple positive pairs for {scene_id}/{evidence_id}")
        coverage[(rows[0]["split"], rows[0]["role"])].append(bool(positive))

    groups_by_scene: dict[str, list[list[dict[str, Any]]]] = defaultdict(list)
    for (scene_id, _), rows in groups.items():
        groups_by_scene[scene_id].append(rows)
    for scene_id, scene_groups in groups_by_scene.items():
        hard_ids = [
            next(
                row["candidate_id"] for row in rows
                if "global_hard_assignment" in row["proposal"]["proposal_sources"]
            )
            for rows in scene_groups
        ]
        source_candidate_count = len(scene_by_id[scene_id]["agent_input"]["atom_candidates"])
        if len(scene_groups) == source_candidate_count and len(set(hard_ids)) == source_candidate_count:
            perfect_matching_scenes += 1
        else:
            errors.append(f"Hard proposal edges do not form a perfect matching in {scene_id}")

    fit_eval_overlap = prompts["grounding_fit"] & prompts["grounding_evaluation"]
    gates = config["gates"]["preExtraction"]
    coverage_metrics = {
        split: {
            role: sum(coverage[(split, role)]) / len(coverage[(split, role)])
            for role in ("support", "query") if coverage[(split, role)]
        }
        for split in ("grounding_fit", "grounding_calibration", "grounding_evaluation")
    }
    if coverage_metrics["grounding_evaluation"]["support"] < gates[
        "minimumEvaluationSupportGoldProposalCoverage"
    ]:
        errors.append("Evaluation-support gold proposal coverage misses the registered gate")
    if coverage_metrics["grounding_evaluation"]["query"] < gates[
        "minimumEvaluationQueryGoldProposalCoverage"
    ]:
        errors.append("Evaluation-query gold proposal coverage misses the registered gate")
    if len(fit_eval_overlap) > gates["maximumExactFitEvaluationPromptOverlap"]:
        errors.append("Exact fit/evaluation cross prompts overlap")
    if len(pairs) > gates["maximumNewModelForwardPasses"]:
        errors.append("Pair corpus exceeds the registered model-forward budget")

    ids = [row["id"] for row in pairs]
    if len(set(ids)) != len(ids):
        errors.append("Pair IDs are not unique")
    ordered = sorted(pairs, key=lambda row: row["id"])
    corpus_hash = sha256_text("".join(canonical_json(row) + "\n" for row in ordered))
    if manifest["corpus_sha256"] != corpus_hash:
        errors.append("Materialized pair corpus differs from its manifest")
    if manifest["pairs"] != len(pairs):
        errors.append("Manifest pair count differs from the materialized corpus")
    if manifest["config_sha256"] != file_sha256(PROJECT_ROOT / manifest["config"]):
        errors.append("V24 configuration differs from the proposal manifest")
    for key, expected in manifest["source_hashes"].items():
        actual = file_sha256(PROJECT_ROOT / config[key])
        if actual != expected:
            errors.append(f"Source artifact changed after V24 proposal construction: {key}")

    truth_by_split = {
        split: dict(sorted(Counter(
            row["target"]["truth_label"] for row in pairs
            if row["split"] == split and row["target"]["same_atom"]
        ).items()))
        for split in ("grounding_fit", "grounding_calibration", "grounding_evaluation")
    }
    if set(truth_by_split["grounding_fit"]) != set(config["heads"]["truth"]["classes"]):
        errors.append("Grounding-fit positive pairs do not cover every registered truth class")

    return {
        "schema_version": 24,
        "experiment": "v24_pre_extraction_proposal_and_firewall_audit",
        "passed": not errors,
        "decision": "authorize_v24_protocol_lock" if not errors else "repair_v24_before_model_access",
        "errors": errors,
        "population": {
            "pairs": len(pairs),
            "evidence_groups": len(groups),
            "scenes": len(groups_by_scene),
            "split_counts": dict(sorted(Counter(row["split"] for row in pairs).items())),
            "proposal_sizes": {str(key): value for key, value in sorted(proposal_sizes.items())},
            "pair_class_counts": dict(sorted(Counter(
                row["target"]["pair_class"] for row in pairs
            ).items())),
            "positive_truth_counts_by_split": truth_by_split,
        },
        "proposal": {
            "gold_coverage_by_split_and_role": coverage_metrics,
            "hard_edges_present": hard_edges_present,
            "perfect_matching_scenes": perfect_matching_scenes,
        },
        "firewall": {
            "forbidden_agent_input_keys": dict(sorted(leaked_keys.items())),
            "exact_source_agent_inputs": exact_agent_target_matches,
            "exact_fit_evaluation_prompt_overlap": len(fit_eval_overlap),
            "new_model_forward_passes_before_lock": 0,
            "new_linear_fits_before_lock": 0,
            "fresh_benchmark_records_created": 0,
        },
        "budget": {
            "registered_maximum_model_forwards": gates["maximumNewModelForwardPasses"],
            "planned_model_forwards": len(pairs),
        },
        "integrity": {
            "corpus_sha256": corpus_hash,
            "manifest_source_hashes_verified": len(manifest["source_hashes"]),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/v24-cross-encoder.json")
    parser.add_argument("--output", default="outputs/v24-cross-encoder/pre-extraction-audit.json")
    args = parser.parse_args()
    config = json.loads((PROJECT_ROOT / args.config).read_text())
    proposal_root = PROJECT_ROOT / config["outputDir"]
    pairs = read_pairs(proposal_root)
    manifest = json.loads((proposal_root / "manifest.json").read_text())
    original_lock = json.loads((PROJECT_ROOT / config["sourceV22r2Lock"]).read_text())
    scenes = read_jsonl_directory(PROJECT_ROOT / original_lock["source"]["dataset"] / "scenes")
    prediction_rows = [
        json.loads(line) for line in (PROJECT_ROOT / config["sourcePredictions"]).read_text().splitlines()
        if line.strip()
    ]
    result = audit(pairs, config, manifest, scenes, prediction_rows)
    output = (PROJECT_ROOT / args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
