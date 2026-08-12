"""Structural, semantic, prompt, and firewall audit for materialized V21 data."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from audit_v18_benchmark import read_records
from audit_v19_compatibility import prompt_inventory, read_scenes, recursive_keys, tokenization_audit
from v10_protocol import file_sha256
from v21_final_suite import canonical_json, sha256_text, structural_summary


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN_AGENT_KEYS = {
    "action_dependency_schema", "allowed_values", "assignment", "behavioral_signature",
    "current_value", "executable_schema", "hypothesis_relations", "relevant_determinants",
    "transition_cases", "target",
}


def dataset_sha256(root: Path) -> str:
    names = ("episodes.jsonl", "novel_ontology.jsonl", "supported.jsonl")
    return sha256_text("".join(f"{name}\n{(root / name).read_text()}" for name in names))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lock", default="configs/v21-multimechanic-execution-lock.json")
    parser.add_argument("--dataset", default="data/v21-final")
    parser.add_argument("--output", default="outputs/v21-final/pre-extraction-audit.json")
    args = parser.parse_args()
    lock_path = PROJECT_ROOT / args.lock
    lock = json.loads(lock_path.read_text())
    for path, expected in lock["implementation"].items():
        if file_sha256(PROJECT_ROOT / path) != expected:
            raise RuntimeError(f"V21 locked implementation changed: {path}")
    root = PROJECT_ROOT / args.dataset
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    seed_ledger_path = PROJECT_ROOT / manifest["seed_ledger"]
    seed_ledger = json.loads(seed_ledger_path.read_text())
    artifacts_match = all(
        file_sha256(root / name) == expected
        for name, expected in manifest["artifact_sha256"].items()
    )
    episodes = [json.loads(line) for line in (root / "episodes.jsonl").read_text().splitlines() if line]
    scenes = read_scenes(root)
    summary = structural_summary(episodes, scenes)
    v18_records = read_records(PROJECT_ROOT / lock["source"]["v18_dataset"])
    v18_signatures = {tuple(value["target"]["behavioral_signature"]) for value in v18_records}
    final_signatures = {tuple(value["target"]["behavioral_signature"]) for value in episodes}
    pairing: dict[tuple[str, str, str], dict[str, dict[str, Any]]] = {}
    for scene in scenes:
        key = (scene["episode_id"], scene["item_kind"], scene["source_item_id"])
        pairing.setdefault(key, {})[scene["view"]] = scene
    pair_errors = 0
    for pair in pairing.values():
        if set(pair) != {"supported", "novel_ontology"}:
            pair_errors += 1
            continue
        supported = pair["supported"]
        novel = pair["novel_ontology"]
        if [value["allowed_values"] for value in supported["target"]["determinant_grounding"]] != [
            value["allowed_values"] for value in novel["target"]["determinant_grounding"]
        ]:
            pair_errors += 1
        if supported.get("observed_transition_code") != novel.get("observed_transition_code"):
            pair_errors += 1
    leaked_inputs = [
        value["id"] for value in [*episodes, *scenes]
        if recursive_keys(value["agent_input"]) & FORBIDDEN_AGENT_KEYS
    ]
    base_prompts, nli_prompts, base_evidence = prompt_inventory(scenes)
    prompts = {
        "unique_base_prompts": len(base_prompts),
        "unique_nli_prompts": len(nli_prompts),
        "new_model_forward_passes": len(base_prompts) + len(nli_prompts),
        "base_prompt_sha256": sha256_text(canonical_json(base_prompts)),
        "nli_prompt_sha256": sha256_text(canonical_json(nli_prompts)),
    }
    tokenization = tokenization_audit(base_prompts, nli_prompts, base_evidence, lock["model"])
    expected_families = {
        family: values["episodes"] for family, values in lock["config"]["constructionFamilies"].items()
    }
    checks = {
        "manifest_execution_lock_matches": (
            manifest["execution_lock_sha256"] == file_sha256(lock_path)
        ),
        "manifest_design_lock_matches": (
            manifest["design_lock_sha256"] == lock["design_lock_sha256"]
        ),
        "single_seed_draw": seed_ledger["draw_number"] == 1 and manifest["construction_number"] == 1,
        "seed_ledger_hash_matches": manifest["seed_ledger_sha256"] == file_sha256(seed_ledger_path),
        "seed_matches_manifest": manifest["seed"] == seed_ledger["seed"],
        "artifact_hashes_match": artifacts_match,
        "dataset_hash_matches": manifest["dataset_sha256"] == dataset_sha256(root),
        "exactly_40_mechanics": summary["episodes"] == 40,
        "family_quotas_exact": summary["family_counts"] == expected_families,
        "one_two_bit_balance": summary["outcome_bit_counts"] == {"1": 20, "2": 20},
        "injectivity_balance": summary["injectivity_counts"] == {
            "injective": 20, "non_injective": 20,
        },
        "unique_final_behaviors": len(final_signatures) == 40,
        "zero_v18_behavior_overlap": not (final_signatures & v18_signatures),
        "paired_views_exact": pair_errors == 0,
        "all_surfaces_present": set(summary["surface_counts"]) == set(lock["config"]["language"]["surfaceFamilies"]),
        "all_semantic_operators_present": set(summary["semantic_operator_counts"]) == set(lock["config"]["language"]["semanticOperators"]),
        "all_unresolved_modes_present": set(summary["unresolved_mode_counts"]) == set(lock["config"]["language"]["unresolvedModes"]),
        "zero_agent_input_leaks": not leaked_inputs,
        "prompt_budget_exact": prompts["new_model_forward_passes"] == 5136,
        "prompt_budget_bounded": (
            prompts["new_model_forward_passes"] <= lock["limits"]["maximumNewModelForwardPasses"]
        ),
        "token_spans_nonempty": tokenization["empty_target_spans"] == 0,
        "no_prompt_truncation": tokenization["truncated_prompts"] == 0,
        "features_absent": not (PROJECT_ROOT / "outputs/v21-final/features").exists(),
        "evaluation_absent": not (PROJECT_ROOT / "outputs/v21-final/evaluation").exists(),
        "no_retry": lock["limits"]["retriesPermitted"] == 0,
    }
    report = {
        "schema_version": 21,
        "experiment": "v21_final_pre_extraction_audit",
        "execution_lock": args.lock,
        "execution_lock_sha256": file_sha256(lock_path),
        "dataset": args.dataset,
        "manifest_sha256": file_sha256(manifest_path),
        "dataset_sha256": manifest["dataset_sha256"],
        "summary": summary,
        "prompt_inventory": prompts,
        "tokenization": tokenization,
        "v18_behavior_overlap": len(final_signatures & v18_signatures),
        "pair_errors": pair_errors,
        "leaked_agent_inputs": leaked_inputs,
        "checks": checks,
        "passed": all(checks.values()),
        "decision": "authorize_single_v21_feature_extraction" if all(checks.values()) else "block_v21_extraction",
        "data_access": {
            "final_records_read": len(episodes),
            "final_scenes_read": len(scenes),
            "final_labels_used_for_model_selection": 0,
            "model_forward_passes": 0,
            "new_linear_fits": 0,
            "adapter_training_runs": 0,
            "v17_records_read": 0,
            "v17_model_results_read": 0,
        },
    }
    output = PROJECT_ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
