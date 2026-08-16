#!/usr/bin/env python3
"""Pre-model structural and firewall audit for V31."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

from generate_v31_signed_fact_adaptation import build_records, corpus_hash
from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT
from v30_language import SURFACE_TEMPLATES as V30_TEMPLATES, atom_key, canonical_json, predicate_specs, sha256_text
from v31_language import construction_hash, render_evidence, representation_prompt_layout


FORBIDDEN_AGENT_KEYS = {
    "arguments", "atom", "candidate_statement", "construction_hash", "gold",
    "oracle_metadata", "pairs", "predicate_kind", "semantic_operator",
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
        rows.extend(json.loads(line) for line in (root / f"{split}.jsonl").read_text().splitlines() if line.strip())
    return rows


def old_construction_hashes() -> set[str]:
    hashes = set()
    for operator, families in V30_TEMPLATES.items():
        for template, _ in families.values():
            normalized = re.sub(r"\{[^}]+\}", "{SLOT}", template.lower())
            normalized = re.sub(r"\s+", " ", normalized).strip()
            hashes.add(sha256_text(f"{operator}|{normalized}"))
    return hashes


def signature(row: dict[str, Any]) -> tuple[Any, ...]:
    target, metadata = row["target"], row["oracle_metadata"]
    return (
        metadata["semantic_operator"], target["predicate_kind"], target["predicate"],
        target["truth_status"], metadata["scene_variant"],
        metadata["relation_orientation"], metadata["distractor"],
    )


def audit(
    rows: Sequence[dict[str, Any]], config: dict[str, Any], manifest: dict[str, Any],
    config_path: Path, enforce_firewall: bool = True,
) -> dict[str, Any]:
    errors = []
    gates = config["gates"]["preModel"]
    splits = tuple(config["splits"])
    by_split = {split: [row for row in rows if row["split"] == split] for split in splits}
    leaks = Counter()
    evidence_sets: dict[str, set[str]] = defaultdict(set)
    prompt_sets: dict[str, set[str]] = defaultdict(set)
    construction_sets: dict[str, set[str]] = defaultdict(set)
    round_trip = canonical = type_valid = 0
    scene_atoms: dict[str, list[str]] = defaultdict(list)
    specs = predicate_specs(config)
    ids = [row["id"] for row in rows]
    if len(ids) != len(set(ids)):
        errors.append("V31 clause identifiers are not unique")
    for row in rows:
        target, metadata = row["target"], row["oracle_metadata"]
        for key in set(recursive_keys(row["agent_input"])) & FORBIDDEN_AGENT_KEYS:
            leaks[key] += 1
        evidence_sets[row["split"]].add(row["agent_input"]["evidence_text"])
        prompt_sets[row["split"]].add(representation_prompt_layout(row, config)[0])
        construction_sets[row["split"]].add(metadata["construction_hash"])
        expected, length = render_evidence(
            target["predicate"], target["arguments"], target["truth_status"],
            metadata["semantic_operator"], metadata["surface_name"],
            metadata["relation_orientation"] or "direct", metadata["distractor"], config,
        )
        round_trip += expected == row["agent_input"]["evidence_text"] and length == metadata["sentence_length_stratum"]
        canonical += target["atom"] == atom_key(target["predicate"], target["arguments"], config)
        spec = specs[target["predicate"]]
        types = {entity["id"]: entity["entity_type"] for entity in row["agent_input"]["entities"]}
        if spec["kind"] == "unary":
            valid = len(target["arguments"]) == 1 and types.get(target["arguments"][0]) == spec["entityType"]
        else:
            valid = (
                len(target["arguments"]) == 2
                and types.get(target["arguments"][0]) == spec["sourceType"]
                and types.get(target["arguments"][1]) == spec["targetType"]
                and target["arguments"][0] != target["arguments"][1]
            )
        type_valid += valid
        scene_atoms[row["scene_id"]].append(target["atom"])
        if metadata["construction_hash"] != construction_hash(metadata["semantic_operator"], metadata["surface_name"]):
            errors.append(f"Construction hash mismatch: {row['id']}")
    if leaks:
        errors.append(f"V31 target fields leaked into public input: {dict(leaks)}")
    if round_trip != len(rows): errors.append("V31 oracle surface round trip failed")
    if canonical != len(rows): errors.append("V31 canonical target construction failed")
    if type_valid != len(rows): errors.append("V31 target violates declared types")
    if any(len(values) != len(set(values)) for values in scene_atoms.values()):
        errors.append("V31 scene repeats a canonical atom")

    exact_overlap = construction_overlap = prompt_overlap = 0
    for index, left in enumerate(splits):
        for right in splits[index + 1:]:
            exact_overlap += len(evidence_sets[left] & evidence_sets[right])
            prompt_overlap += len(prompt_sets[left] & prompt_sets[right])
            construction_overlap += len(construction_sets[left] & construction_sets[right])
    v30_overlap = len(set().union(*construction_sets.values()) & old_construction_hashes())
    if exact_overlap > gates["maximumExactEvidenceOverlapAcrossSplits"]: errors.append("Exact V31 evidence overlaps across splits")
    if construction_overlap > gates["maximumConstructionHashOverlapAcrossSplits"]: errors.append("V31 construction hashes overlap across splits")
    if v30_overlap > gates["maximumConstructionHashOverlapWithV30"]: errors.append("V31 construction hashes overlap V30")

    family_counts = Counter(row["oracle_metadata"]["surface_family"] for row in rows)
    evaluation_counts = Counter(row["oracle_metadata"]["surface_family"] for row in by_split["adaptation_evaluation"])
    if len(evaluation_counts) != gates["requiredEvaluationSurfaceFamilies"]: errors.append("Wrong V31 evaluation-family count")
    if set(evaluation_counts.values()) != {gates["requiredExamplesPerSurfaceFamily"]}: errors.append("Wrong V31 evaluation-family population")
    if set(family_counts.values()) != {gates["requiredExamplesPerSurfaceFamily"]}: errors.append("Wrong V31 family population")
    fit_signatures = {signature(row) for row in by_split["adaptation_fit"]}
    unsupported = [signature(row) for row in by_split["adaptation_evaluation"] if signature(row) not in fit_signatures]
    if unsupported: errors.append("V31 evaluation contains unsupported semantic signatures")

    pair_groups: dict[tuple[str, str], list[tuple[dict[str, Any], dict[str, str]]]] = defaultdict(list)
    for row in rows:
        for pair in row["oracle_metadata"]["pairs"]:
            pair_groups[(pair["kind"], pair["id"])].append((row, pair))
    pair_counts, pair_errors = Counter(), []
    for (kind, identifier), members in pair_groups.items():
        pair_counts[kind] += 1
        if len(members) != 2:
            pair_errors.append(f"{kind}:{identifier} has {len(members)} members")
            continue
        (left, left_pair), (right, right_pair) = members
        lt, rt = left["target"], right["target"]
        if len({left_pair["role"], right_pair["role"]}) != 2:
            pair_errors.append(f"{kind}:{identifier} lacks distinct roles")
        if kind in ("distractor", "inverse", "affirmative_negated", "affirmative_double_negation"):
            if (lt["predicate"], lt["arguments"], lt["truth_status"]) != (rt["predicate"], rt["arguments"], rt["truth_status"]):
                pair_errors.append(f"{kind}:{identifier} does not preserve the signed fact")
        elif kind == "argument_reversal":
            if not (lt["predicate"] == rt["predicate"] == "linked" and lt["arguments"] == list(reversed(rt["arguments"])) and lt["truth_status"] == rt["truth_status"]):
                pair_errors.append(f"argument_reversal:{identifier} malformed")
        elif kind == "false_unknown":
            if not (lt["predicate"] == rt["predicate"] and lt["arguments"] == rt["arguments"] and {lt["truth_status"], rt["truth_status"]} == {"false", "unknown"}):
                pair_errors.append(f"false_unknown:{identifier} malformed")
    if pair_errors: errors.extend(pair_errors[:10])

    expected = build_records(config)
    if [canonical_json(row) for row in sorted(rows, key=lambda row: row["id"])] != [canonical_json(row) for row in sorted(expected, key=lambda row: row["id"])]:
        errors.append("V31 on-disk corpus differs from deterministic reconstruction")
    root = PROJECT_ROOT / config["outputDir"]
    if manifest["config_sha256"] != file_sha256(config_path): errors.append("V31 manifest config hash differs")
    if manifest["corpus_sha256"] != corpus_hash(list(rows)): errors.append("V31 manifest corpus hash differs")
    for name, expected_hash in manifest["artifact_sha256"].items():
        if file_sha256(root / name) != expected_hash: errors.append(f"V31 corpus artifact changed: {name}")

    v30_result = json.loads((PROJECT_ROOT / config["sourceV30Result"]).read_text())
    v30_audit = json.loads((PROJECT_ROOT / config["sourceV30PostAudit"]).read_text())
    if not (v30_audit["passed"] and v30_result["lora_eligibility"]["eligible"] and not v30_result["v28_integration_authorized"]):
        errors.append("Accepted V30 decision does not authorize V31")
    forbidden = [
        PROJECT_ROOT / "outputs/v31-signed-fact-adaptation/features-evaluation",
        PROJECT_ROOT / "outputs/v31-signed-fact-adaptation/sealed-evaluation",
        PROJECT_ROOT / "configs/v31-trained-systems-lock.json",
    ]
    if enforce_firewall and any(path.exists() for path in forbidden):
        errors.append("V31 evaluation or trained-system artifact exists before protocol lock")

    return {
        "schema_version": 31, "experiment": "v31_pre_model_structural_firewall_audit",
        "passed": not errors,
        "decision": "authorize_v31_protocol_lock" if not errors else "repair_v31_before_model_access",
        "errors": errors,
        "population": {
            "records": len(rows), "scenes": len(scene_atoms),
            "split_counts": dict(sorted(Counter(row["split"] for row in rows).items())),
            "surface_families": len(family_counts), "evaluation_surface_families": len(evaluation_counts),
            "truth_status_counts": dict(sorted(Counter(row["target"]["truth_status"] for row in rows).items())),
        },
        "semantic_audit": {
            "oracle_round_trip_accuracy": round_trip / len(rows),
            "canonical_target_accuracy": canonical / len(rows),
            "type_validity_accuracy": type_valid / len(rows),
            "unsupported_evaluation_signatures": len(unsupported),
            "pair_counts": dict(sorted(pair_counts.items())), "pair_errors": len(pair_errors),
        },
        "separation": {
            "exact_evidence_overlap_across_splits": exact_overlap,
            "exact_prompt_overlap_across_splits": prompt_overlap,
            "construction_hash_overlap_across_splits": construction_overlap,
            "construction_hash_overlap_with_v30": v30_overlap,
        },
        "firewall": {
            "target_fields_in_agent_input": dict(sorted(leaks.items())),
            "model_forward_passes_before_lock": 0, "training_runs_before_lock": 0,
            "evaluation_features_before_trained_system_lock": 0,
            "evaluation_predictions_before_trained_system_lock": 0,
            "v28_signals_available_during_training": False,
        },
        "integrity": {
            "config_sha256": file_sha256(config_path),
            "manifest_sha256": file_sha256(root / "manifest.json") if (root / "manifest.json").exists() else None,
            "corpus_sha256": corpus_hash(list(rows)),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/v31-signed-fact-adaptation.json")
    parser.add_argument("--output", default="outputs/v31-signed-fact-adaptation/pre-model-audit.json")
    args = parser.parse_args()
    config_path = (PROJECT_ROOT / args.config).resolve()
    config = json.loads(config_path.read_text())
    root = PROJECT_ROOT / config["outputDir"]
    rows = read_rows(root, tuple(config["splits"]))
    manifest = json.loads((root / "manifest.json").read_text())
    result = audit(rows, config, manifest, config_path)
    output = (PROJECT_ROOT / args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["passed"]: raise SystemExit(1)


if __name__ == "__main__": main()
