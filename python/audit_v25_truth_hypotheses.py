"""Pre-extraction structural, semantic, and firewall audit for V25."""

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
from v24_cross_encoder import sha256_text
from v25_truth_hypotheses import truth_prompt_text


FORBIDDEN_PUBLIC_KEYS = {
    "allowed_values", "atom", "atom_groundings", "candidate_assignment_correct",
    "compatible", "epistemic_state", "possible_transition_codes", "program",
    "query_axis", "reference_complete_world", "semantic_operator", "truth_label",
    "use_for_fit",
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


def audit(
    rows: Sequence[dict[str, Any]], config: dict[str, Any], manifest: dict[str, Any],
    scenes: Sequence[dict[str, Any]], predictions: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    errors: list[str] = []
    scene_lookup = {row["id"]: row for row in scenes}
    prediction_lookup = {row["scene_id"]: row for row in predictions}
    hypotheses = {row["id"]: row for row in config["assessmentHypotheses"]}
    base_groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    fixed_groups: dict[tuple[str, str], set[str]] = defaultdict(set)
    prompts: dict[str, set[str]] = defaultdict(set)
    leaks = Counter()
    exact_agent_inputs = 0
    for row in rows:
        base_groups[(row["scene_id"], row["evidence_id"], row["candidate_id"])].append(row)
        if "v24_fixed_assignment" in row["selection_sources"]:
            fixed_groups[(row["scene_id"], row["evidence_id"])].add(row["candidate_id"])
        prompts[row["split"]].add(truth_prompt_text(row))
        for key in set(recursive_keys(row["agent_input"])) & FORBIDDEN_PUBLIC_KEYS:
            leaks[key] += 1
        scene = scene_lookup[row["scene_id"]]
        public = scene["agent_input"]
        evidence = {value["id"]: value["text"] for value in public["evidence"]}
        candidates = {value["id"]: value["statement"] for value in public["atom_candidates"]}
        expected = {
            "entities": public["entities"],
            "action": public["action"],
            "evidence_text": evidence[row["evidence_id"]],
            "candidate_statement": candidates[row["candidate_id"]],
            "assessment_statement": hypotheses[row["assessment_id"]]["statement"],
            "instruction": "Represent whether the assessment hypothesis fits the evidence and candidate fact.",
        }
        exact_agent_inputs += row["agent_input"] == expected
    if leaks:
        errors.append(f"Target/oracle fields leaked into V25 prompts: {dict(leaks)}")
    if exact_agent_inputs != len(rows):
        errors.append("One or more V25 agent inputs differ from registered source fields")

    expected_ids = set(hypotheses)
    for key, group in base_groups.items():
        if len(group) != config["gates"]["preExtraction"]["requiredAssessmentHypothesesPerPair"]:
            errors.append(f"Assessment cardinality differs for {key}")
        if {row["assessment_id"] for row in group} != expected_ids:
            errors.append(f"Assessment inventory differs for {key}")
        if sum(row["target"]["compatible"] for row in group) != 1:
            errors.append(f"V25 base pair does not have exactly one compatible assessment: {key}")

    all_evidence = {
        (scene["id"], evidence["id"])
        for scene in scenes for evidence in scene["agent_input"]["evidence"]
    }
    if set(fixed_groups) != all_evidence:
        errors.append("V25 fixed-assignment rows do not cover every evidence unit")
    fixed_correct = 0
    for scene_id, evidence_id in all_evidence:
        source = {
            row["evidence_id"]: row for row in prediction_lookup[scene_id]["rows"]
        }[evidence_id]["candidate_id"]
        if fixed_groups[(scene_id, evidence_id)] == {source}:
            fixed_correct += 1
        else:
            errors.append(f"V25 changed the V24 fixed assignment for {scene_id}/{evidence_id}")

    fit_gold_groups = 0
    fit_training_rows = []
    for scene in scenes:
        if scene["split"] != config["head"]["fitSplit"]:
            continue
        target = {row["evidence_id"]: row for row in scene["target"]["atom_groundings"]}
        for evidence_id, target_row in target.items():
            group = base_groups.get((scene["id"], evidence_id, target_row["candidate_id"]), [])
            if len(group) == 3 and all(row["target"]["use_for_fit"] for row in group):
                fit_gold_groups += 1
                fit_training_rows.extend(group)
            else:
                errors.append(f"Clean V25 fit triple is missing for {scene['id']}/{evidence_id}")
    invalid_fit = [
        row for row in rows if row["target"]["use_for_fit"]
        and not (
            row["split"] == config["head"]["fitSplit"]
            and "fit_gold_candidate" in row["selection_sources"]
        )
    ]
    if invalid_fit:
        errors.append("V25 fit flag occurs outside a clean fit gold-candidate triple")

    overlap = prompts["grounding_fit"] & prompts["grounding_evaluation"]
    gates = config["gates"]["preExtraction"]
    if len(overlap) > gates["maximumExactFitEvaluationPromptOverlap"]:
        errors.append("Exact V25 fit/evaluation prompts overlap")
    if len(rows) > gates["maximumNewModelForwardPasses"]:
        errors.append("V25 row corpus exceeds the registered forward budget")
    fixed_coverage = len(fixed_groups) / len(all_evidence)
    if fixed_coverage != gates["requiredFixedAssignmentCoverage"]:
        errors.append("V25 fixed-assignment coverage differs from registration")

    ordered = sorted(rows, key=lambda row: row["id"])
    corpus_hash = sha256_text("".join(canonical_json(row) + "\n" for row in ordered))
    if manifest["corpus_sha256"] != corpus_hash or manifest["rows"] != len(rows):
        errors.append("V25 materialized corpus differs from its manifest")
    if manifest["config_sha256"] != file_sha256(PROJECT_ROOT / manifest["config"]):
        errors.append("V25 config differs from its manifest")
    for key, expected in manifest["source_hashes"].items():
        if file_sha256(PROJECT_ROOT / config[key]) != expected:
            errors.append(f"V25 source artifact changed after construction: {key}")

    post_audit = json.loads((PROJECT_ROOT / config["sourceV24PostAudit"]).read_text())
    result = json.loads((PROJECT_ROOT / config["sourceV24Result"]).read_text())
    diagnostic = json.loads((PROJECT_ROOT / config["sourceV24Diagnostic"]).read_text())
    if not post_audit["passed"] or post_audit["decision"] != "accept_v24_exposed_development_result":
        errors.append("V24 integrity status does not authorize V25")
    if result["decision"] != "factor_truth_semantics_before_fresh_benchmark_no_lora":
        errors.append("V24 registered decision does not authorize V25")
    if diagnostic["data_access"]["new_model_forward_passes"] != 0 or diagnostic["data_access"]["new_linear_fits"] != 0:
        errors.append("V24 diagnostic reports unregistered new model work")

    return {
        "schema_version": 25,
        "experiment": "v25_pre_extraction_truth_hypothesis_audit",
        "passed": not errors,
        "decision": "authorize_v25_protocol_lock" if not errors else "repair_v25_before_model_access",
        "errors": errors,
        "population": {
            "rows": len(rows),
            "base_pairs": len(base_groups),
            "evidence_groups": len(all_evidence),
            "fixed_assignment_groups": len(fixed_groups),
            "fit_gold_groups": fit_gold_groups,
            "fit_training_rows": len(fit_training_rows),
            "fit_positive_rows": sum(row["target"]["compatible"] for row in fit_training_rows),
            "split_counts": dict(sorted(Counter(row["split"] for row in rows).items())),
            "assessment_counts": dict(sorted(Counter(row["assessment_id"] for row in rows).items())),
        },
        "fixed_assignment": {
            "coverage": fixed_coverage,
            "exact_v24_assignment_groups": fixed_correct,
        },
        "firewall": {
            "forbidden_agent_input_keys": dict(sorted(leaks.items())),
            "exact_source_agent_inputs": exact_agent_inputs,
            "exact_fit_evaluation_prompt_overlap": len(overlap),
            "new_model_forward_passes_before_lock": 0,
            "new_linear_fits_before_lock": 0,
            "fresh_benchmark_records_created": 0,
        },
        "budget": {
            "planned_model_forwards": len(rows),
            "registered_maximum_model_forwards": gates["maximumNewModelForwardPasses"],
        },
        "integrity": {
            "corpus_sha256": corpus_hash,
            "manifest_source_hashes_verified": len(manifest["source_hashes"]),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/v25-truth-hypotheses.json")
    parser.add_argument("--output", default="outputs/v25-truth-hypotheses/pre-extraction-audit.json")
    args = parser.parse_args()
    config = json.loads((PROJECT_ROOT / args.config).read_text())
    root = PROJECT_ROOT / config["outputDir"]
    rows = read_rows(root)
    manifest = json.loads((root / "manifest.json").read_text())
    v24_lock = json.loads((PROJECT_ROOT / config["sourceV24Lock"]).read_text())
    original_lock = json.loads((PROJECT_ROOT / v24_lock["source"]["v22r2_lock"]).read_text())
    scenes = read_jsonl_directory(PROJECT_ROOT / original_lock["source"]["dataset"] / "scenes")
    predictions = [
        json.loads(line) for line in (PROJECT_ROOT / config["sourceV24Predictions"]).read_text().splitlines()
        if line.strip()
    ]
    result = audit(rows, config, manifest, scenes, predictions)
    output = (PROJECT_ROOT / args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
