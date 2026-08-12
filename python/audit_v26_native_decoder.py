"""Pre-evaluation integrity and firewall audit for V26 native decoding."""

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
from v26_native_decoder import decoder_prompt


FORBIDDEN_PUBLIC_KEYS = {
    "allowed_values", "atom", "atom_groundings", "candidate_assignment_correct",
    "epistemic_state", "possible_transition_codes", "program", "query_axis",
    "reference_complete_world", "semantic_operator", "truth_label",
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
    errors = []
    scene_lookup = {row["id"]: row for row in scenes}
    prediction_lookup = {row["scene_id"]: row for row in predictions}
    prompts = defaultdict(set)
    leaks = Counter()
    exact_inputs = 0
    fixed_exact = 0
    seen = set()
    for row in rows:
        key = (row["scene_id"], row["evidence_id"])
        if key in seen:
            errors.append(f"Repeated V26 evidence row: {key}")
        seen.add(key)
        prompts[row["split"]].add(decoder_prompt(row))
        for leaked in set(recursive_keys(row["agent_input"])) & FORBIDDEN_PUBLIC_KEYS:
            leaks[leaked] += 1
        scene = scene_lookup[row["scene_id"]]
        public = scene["agent_input"]
        evidence = {value["id"]: value["text"] for value in public["evidence"]}
        candidates = {value["id"]: value["statement"] for value in public["atom_candidates"]}
        expected = {
            "entities": public["entities"],
            "action": public["action"],
            "evidence_text": evidence[row["evidence_id"]],
            "candidate_statement": candidates[row["candidate_id"]],
            "instruction": "Classify the evidence relative to the candidate fact using the registered letter code.",
        }
        exact_inputs += row["agent_input"] == expected
        source_candidate = {
            value["evidence_id"]: value for value in prediction_lookup[row["scene_id"]]["rows"]
        }[row["evidence_id"]]["candidate_id"]
        fixed_exact += row["candidate_id"] == source_candidate
    if leaks:
        errors.append(f"Target/oracle fields leaked into V26 prompts: {dict(leaks)}")
    if exact_inputs != len(rows):
        errors.append("V26 inputs differ from registered source fields")
    if fixed_exact != len(rows):
        errors.append("V26 changes one or more fixed V24 assignments")

    all_evidence = {
        (scene["id"], evidence["id"])
        for scene in scenes for evidence in scene["agent_input"]["evidence"]
    }
    if seen != all_evidence:
        errors.append("V26 does not cover exactly the V22r2 evidence population")
    gates = config["gates"]["preEvaluation"]
    overlap = prompts["grounding_fit"] & prompts["grounding_evaluation"]
    if len(overlap) > gates["maximumExactFitEvaluationPromptOverlap"]:
        errors.append("Exact V26 fit/evaluation prompts overlap")
    if len(rows) > gates["maximumNewModelForwardPasses"]:
        errors.append("V26 exceeds the registered forward budget")
    labels = config["labels"]
    if len(labels) != gates["requiredSingleTokenLabels"] or [row["token"] for row in labels] != ["A", "B", "C"]:
        errors.append("V26 label inventory differs from registration")
    if len({row["truthLabel"] for row in labels}) != 3:
        errors.append("V26 labels do not map one-to-one to truth statuses")

    ordered = sorted(rows, key=lambda row: row["id"])
    corpus_hash = sha256_text("".join(canonical_json(row) + "\n" for row in ordered))
    if manifest["rows"] != len(rows) or manifest["corpus_sha256"] != corpus_hash:
        errors.append("V26 corpus differs from its manifest")
    if manifest["config_sha256"] != file_sha256(PROJECT_ROOT / manifest["config"]):
        errors.append("V26 config differs from its manifest")
    for key, expected in manifest["source_hashes"].items():
        if file_sha256(PROJECT_ROOT / config[key]) != expected:
            errors.append(f"V26 source changed after construction: {key}")
    v25_audit = json.loads((PROJECT_ROOT / config["sourceV25PostAudit"]).read_text())
    v25_result = json.loads((PROJECT_ROOT / config["sourceV25Result"]).read_text())
    diagnostic = json.loads((PROJECT_ROOT / config["sourceV25Diagnostic"]).read_text())
    if not v25_audit["passed"] or v25_audit["decision"] != "accept_v25_exposed_development_result":
        errors.append("V25 integrity status does not authorize V26")
    if v25_result["decision"] != "explicit_truth_hypotheses_insufficient_no_lora":
        errors.append("V25 registered decision does not authorize V26")
    if not diagnostic["localization"]["reject_more_layer8_linear_head_variants"]:
        errors.append("V25 diagnostic does not justify native decoder pivot")
    return {
        "schema_version": 26,
        "experiment": "v26_pre_evaluation_native_decoder_audit",
        "passed": not errors,
        "decision": "authorize_v26_protocol_lock" if not errors else "repair_v26_before_model_access",
        "errors": errors,
        "population": {
            "rows": len(rows),
            "scenes": len(scenes),
            "split_counts": dict(sorted(Counter(row["split"] for row in rows).items())),
            "role_counts": dict(sorted(Counter(row["role"] for row in rows).items())),
        },
        "fixed_assignment": {
            "coverage": len(seen) / len(all_evidence),
            "exact_v24_assignments": fixed_exact,
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
    parser.add_argument("--config", default="configs/v26-native-truth-decoder.json")
    parser.add_argument("--output", default="outputs/v26-native-truth-decoder/pre-evaluation-audit.json")
    args = parser.parse_args()
    config = json.loads((PROJECT_ROOT / args.config).read_text())
    root = PROJECT_ROOT / config["outputDir"]
    rows = read_rows(root)
    manifest = json.loads((root / "manifest.json").read_text())
    v25_lock = json.loads((PROJECT_ROOT / config["sourceV25Lock"]).read_text())
    v24_lock = json.loads((PROJECT_ROOT / v25_lock["source"]["v24_lock"]).read_text())
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
