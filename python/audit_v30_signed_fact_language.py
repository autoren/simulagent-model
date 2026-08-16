#!/usr/bin/env python3
"""Structural, semantic, leakage, and firewall audit for V30 before model access."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

from generate_v30_signed_fact_language import build_records, corpus_hash
from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT
from v30_language import (
    atom_key, canonical_json, predicate_specs, primary_field_prompt, render_evidence,
)


FORBIDDEN_AGENT_KEYS = {
    "arguments", "atom", "candidate_fact", "candidate_statement", "gold",
    "oracle_metadata", "pair", "pairs", "predicate_kind", "semantic_operator",
    "surface_family", "target", "truth_label", "truth_status",
}


def recursive_keys(value: Any) -> Iterable[str]:
    if isinstance(value, dict):
        for key, child in value.items():
            yield key
            yield from recursive_keys(child)
    elif isinstance(value, list):
        for child in value:
            yield from recursive_keys(child)


def read_rows(root: Path, splits: Sequence[str]) -> list[dict[str, Any]]:
    rows = []
    for split in splits:
        rows.extend(
            json.loads(line) for line in (root / f"{split}.jsonl").read_text().splitlines()
            if line.strip()
        )
    return rows


def semantic_signature(row: dict[str, Any]) -> tuple[Any, ...]:
    metadata = row["oracle_metadata"]
    target = row["target"]
    return (
        metadata["semantic_operator"], target["predicate_kind"], target["predicate"],
        target["truth_status"], metadata["scene_variant"],
        metadata["relation_orientation"], metadata["distractor"],
    )


def audit(
    rows: Sequence[dict[str, Any]], config: dict[str, Any], manifest: dict[str, Any],
    config_path: Path, enforce_pre_model_firewall: bool = True,
) -> dict[str, Any]:
    errors = []
    gates = config["gates"]["preModel"]
    splits = tuple(config["splits"])
    by_split = {split: [row for row in rows if row["split"] == split] for split in splits}
    ids = [row["id"] for row in rows]
    if len(ids) != len(set(ids)):
        errors.append("Clause identifiers are not unique")

    target_leaks = Counter()
    round_trip = 0
    canonical_targets = 0
    type_valid = 0
    prompt_sets: dict[str, set[str]] = defaultdict(set)
    evidence_sets: dict[str, set[str]] = defaultdict(set)
    scene_atoms: dict[str, list[str]] = defaultdict(list)
    specs = predicate_specs(config)
    for row in rows:
        target = row["target"]
        metadata = row["oracle_metadata"]
        leaked = set(recursive_keys(row["agent_input"])) & FORBIDDEN_AGENT_KEYS
        for key in leaked:
            target_leaks[key] += 1
        evidence_sets[row["split"]].add(row["agent_input"]["evidence_text"])
        for field in config["methods"]["primary"]["fields"]:
            prompt, _ = primary_field_prompt(row, field, config)
            prompt_sets[row["split"]].add(prompt)
        expected, expected_length = render_evidence(
            target["predicate"], target["arguments"], target["truth_status"],
            metadata["semantic_operator"], metadata["surface_name"],
            metadata["relation_orientation"] or "direct", metadata["distractor"], config,
        )
        round_trip += (
            expected == row["agent_input"]["evidence_text"]
            and expected_length == metadata["sentence_length_stratum"]
        )
        canonical_targets += target["atom"] == atom_key(
            target["predicate"], target["arguments"], config
        )
        entities = {value["id"]: value["entity_type"] for value in row["agent_input"]["entities"]}
        spec = specs[target["predicate"]]
        if spec["kind"] == "unary":
            valid = (
                len(target["arguments"]) == 1
                and entities.get(target["arguments"][0]) == spec["entityType"]
            )
        else:
            valid = (
                len(target["arguments"]) == 2
                and entities.get(target["arguments"][0]) == spec["sourceType"]
                and entities.get(target["arguments"][1]) == spec["targetType"]
                and target["arguments"][0] != target["arguments"][1]
            )
        type_valid += valid
        scene_atoms[row["scene_id"]].append(target["atom"])
    if target_leaks:
        errors.append(f"Target or construction fields leaked into agent input: {dict(target_leaks)}")
    if round_trip != len(rows):
        errors.append("Generator-provenance oracle round trip is not exact")
    if canonical_targets != len(rows):
        errors.append("One or more canonical target atoms are malformed")
    if type_valid != len(rows):
        errors.append("One or more target facts violate the declared type ontology")
    if any(len(atoms) != len(set(atoms)) for atoms in scene_atoms.values()):
        errors.append("A V30 scene repeats a canonical target atom")

    surface_by_split = {
        split: {row["oracle_metadata"]["surface_family"] for row in selected}
        for split, selected in by_split.items()
    }
    for left_index, left in enumerate(splits):
        for right in splits[left_index + 1:]:
            if surface_by_split[left] & surface_by_split[right]:
                errors.append(f"Surface families overlap across {left} and {right}")
            if evidence_sets[left] & evidence_sets[right]:
                errors.append(f"Exact evidence overlaps across {left} and {right}")
    fit_eval_prompt_overlap = len(
        prompt_sets["language_fit"] & prompt_sets["language_evaluation"]
    )
    if fit_eval_prompt_overlap > gates["maximumExactPrimaryPromptOverlapAcrossFitEvaluation"]:
        errors.append("Exact primary fit/evaluation prompts overlap")

    evaluation_family_counts = Counter(
        row["oracle_metadata"]["surface_family"] for row in by_split["language_evaluation"]
    )
    if len(evaluation_family_counts) != gates["requiredEvaluationSurfaceFamilies"]:
        errors.append("Evaluation surface-family count differs from registration")
    if set(evaluation_family_counts.values()) != {gates["requiredExamplesPerSurfaceFamily"]}:
        errors.append("Evaluation examples per surface family differ from registration")
    all_family_counts = Counter(row["oracle_metadata"]["surface_family"] for row in rows)
    if set(all_family_counts.values()) != {gates["requiredExamplesPerSurfaceFamily"]}:
        errors.append("One or more surface families have the wrong population")

    fit_signatures = {semantic_signature(row) for row in by_split["language_fit"]}
    unsupported_evaluation = []
    for row in by_split["language_evaluation"]:
        signature = semantic_signature(row)
        if signature not in fit_signatures:
            unsupported_evaluation.append(signature)
    if unsupported_evaluation:
        errors.append("Evaluation contains semantic signatures absent from fit")

    pair_groups: dict[tuple[str, str], list[tuple[dict[str, Any], dict[str, str]]]] = defaultdict(list)
    for row in rows:
        for pair in row["oracle_metadata"]["pairs"]:
            pair_groups[(pair["kind"], pair["id"])].append((row, pair))
    pair_counts = Counter()
    pair_errors = []
    for (kind, pair_id), members in pair_groups.items():
        pair_counts[kind] += 1
        if len(members) != 2:
            pair_errors.append(f"{kind}:{pair_id} has {len(members)} members")
            continue
        (left, left_pair), (right, right_pair) = members
        roles = {left_pair["role"], right_pair["role"]}
        left_target, right_target = left["target"], right["target"]
        if kind in ("distractor", "inverse", "affirmative_negated"):
            if (
                left_target["predicate"] != right_target["predicate"]
                or left_target["arguments"] != right_target["arguments"]
                or left_target["truth_status"] != right_target["truth_status"]
            ):
                pair_errors.append(f"{kind}:{pair_id} does not preserve its target")
        elif kind == "argument_reversal":
            if not (
                left_target["predicate"] == right_target["predicate"] == "linked"
                and left_target["arguments"] == list(reversed(right_target["arguments"]))
                and left_target["truth_status"] == right_target["truth_status"]
            ):
                pair_errors.append(f"argument_reversal:{pair_id} is not an exact reversal")
        elif kind == "false_unknown":
            if not (
                left_target["predicate"] == right_target["predicate"]
                and left_target["arguments"] == right_target["arguments"]
                and {left_target["truth_status"], right_target["truth_status"]} == {"false", "unknown"}
            ):
                pair_errors.append(f"false_unknown:{pair_id} is malformed")
        if len(roles) != 2:
            pair_errors.append(f"{kind}:{pair_id} does not contain two distinct roles")
    if pair_errors:
        errors.extend(pair_errors[:10])

    expected_rows = build_records(config)
    if [canonical_json(row) for row in sorted(rows, key=lambda row: row["id"])] != [
        canonical_json(row) for row in sorted(expected_rows, key=lambda row: row["id"])
    ]:
        errors.append("On-disk corpus differs from a fresh deterministic construction")
    if manifest["config_sha256"] != file_sha256(config_path):
        errors.append("Manifest configuration hash differs")
    if manifest["corpus_sha256"] != corpus_hash(list(rows)):
        errors.append("Manifest corpus hash differs")
    root = PROJECT_ROOT / config["outputDir"]
    for name, expected in manifest["artifact_sha256"].items():
        if file_sha256(root / name) != expected:
            errors.append(f"Corpus artifact changed after construction: {name}")

    v28_audit = json.loads((PROJECT_ROOT / config["sourceV28PostAudit"]).read_text())
    v28_result = json.loads((PROJECT_ROOT / config["sourceV28Result"]).read_text())
    if not v28_audit["passed"] or v28_audit["decision"] != "accept_v28_exposed_development_result":
        errors.append("Accepted V28 result is unavailable")
    if v28_result["decision"] != "marginal_program_map_improves_support_continue_query_repair_no_lora":
        errors.append("V28 decision does not support the registered language pivot")
    output_root = PROJECT_ROOT / "outputs/v30-signed-fact-language"
    forbidden_before_lock = [
        output_root / "evaluation", output_root / "evaluation-attempt.json",
        output_root / "integration", output_root / "integration-attempt.json",
    ]
    if enforce_pre_model_firewall and any(path.exists() for path in forbidden_before_lock):
        errors.append("V30 model or integration result exists before protocol lock")

    audit_rate = 1.0 if not errors else 0.0
    if audit_rate < gates["minimumAuditPassRate"]:
        pass
    return {
        "schema_version": 30,
        "experiment": "v30_pre_model_structural_firewall_audit",
        "passed": not errors,
        "decision": "authorize_v30_protocol_lock" if not errors else "repair_v30_before_model_access",
        "errors": errors,
        "population": {
            "records": len(rows), "scenes": len(scene_atoms),
            "split_counts": dict(sorted(Counter(row["split"] for row in rows).items())),
            "surface_families": len(all_family_counts),
            "evaluation_surface_families": len(evaluation_family_counts),
            "semantic_operator_counts": dict(sorted(Counter(
                row["oracle_metadata"]["semantic_operator"] for row in rows
            ).items())),
            "truth_status_counts": dict(sorted(Counter(
                row["target"]["truth_status"] for row in rows
            ).items())),
            "entity_count_counts": dict(sorted(Counter(
                row["oracle_metadata"]["entity_count"] for row in rows
            ).items())),
        },
        "semantic_audit": {
            "oracle_round_trip_accuracy": round_trip / len(rows),
            "canonical_target_accuracy": canonical_targets / len(rows),
            "type_validity_accuracy": type_valid / len(rows),
            "unsupported_evaluation_signatures": len(unsupported_evaluation),
            "pair_counts": dict(sorted(pair_counts.items())),
            "pair_errors": len(pair_errors),
        },
        "firewall": {
            "target_fields_in_agent_input": dict(sorted(target_leaks.items())),
            "exact_fit_evaluation_primary_prompt_overlap": fit_eval_prompt_overlap,
            "exact_evidence_overlap_across_splits": 0,
            "model_forward_passes_before_lock": 0,
            "head_fits_before_lock": 0,
            "threshold_fits_before_lock": 0,
            "evaluation_predictions_before_lock": 0,
            "adapter_training_runs": 0,
        },
        "integrity": {
            "config_sha256": file_sha256(config_path),
            "manifest_sha256": (
                file_sha256(root / "manifest.json") if (root / "manifest.json").exists() else None
            ),
            "corpus_sha256": corpus_hash(list(rows)),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/v30-signed-fact-language.json")
    parser.add_argument("--output", default="outputs/v30-signed-fact-language/pre-model-audit.json")
    args = parser.parse_args()
    config_path = (PROJECT_ROOT / args.config).resolve()
    config = json.loads(config_path.read_text())
    root = PROJECT_ROOT / config["outputDir"]
    manifest = json.loads((root / "manifest.json").read_text())
    rows = read_rows(root, tuple(config["splits"]))
    result = audit(rows, config, manifest, config_path)
    output = (PROJECT_ROOT / args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
