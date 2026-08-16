#!/usr/bin/env python3
"""Structural, semantic, and firewall audit for V32."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

from generate_v32_factorized_semantics import build_records, corpus_hash
from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT
from v30_language import SURFACE_TEMPLATES as V30_TEMPLATES, atom_key, canonical_json, predicate_specs, sha256_text
from v31_language import SURFACE_TEMPLATES as V31_TEMPLATES
from v32_language import compile_truth, construction_hash, render_evidence, representation_prompt_layout


FORBIDDEN_AGENT_KEYS = {
    "arguments", "atom", "candidate_statement", "combination_seen_in_fit",
    "construction_hash", "factorization", "gold", "lexical_sign", "oracle_metadata",
    "outer_operation", "pairs", "predicate_kind", "target", "truth_status",
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


def historical_hashes() -> set[str]:
    result = set()
    for templates in (V30_TEMPLATES, V31_TEMPLATES):
        for operation, families in templates.items():
            for template, _ in families.values():
                normalized = re.sub(r"\{[^}]+\}", "{SLOT}", template.lower())
                normalized = re.sub(r"\s+", " ", normalized).strip()
                result.add(sha256_text(f"{operation}|{normalized}"))
    return result


def audit(
    rows: Sequence[dict[str, Any]], config: dict[str, Any], manifest: dict[str, Any],
    config_path: Path, enforce_firewall: bool = True,
) -> dict[str, Any]:
    errors = []
    splits = tuple(config["splits"])
    gates = config["gates"]["preModel"]
    by_split = {split: [row for row in rows if row["split"] == split] for split in splits}
    leaks = Counter()
    exact_sets, prompt_sets, construction_sets = defaultdict(set), defaultdict(set), defaultdict(set)
    round_trip = canonical = type_valid = compiler_correct = 0
    specs = predicate_specs(config)
    scene_atoms = defaultdict(list)
    for row in rows:
        target, metadata = row["target"], row["oracle_metadata"]
        leaks.update(set(recursive_keys(row["agent_input"])) & FORBIDDEN_AGENT_KEYS)
        exact_sets[row["split"]].add(row["agent_input"]["evidence_text"])
        prompt_sets[row["split"]].add(representation_prompt_layout(row, config)[0])
        construction_sets[row["split"]].add(metadata["construction_hash"])
        factor = target["factorization"]
        expected, length = render_evidence(
            target["predicate"], target["arguments"], factor["lexical_sign"],
            factor["outer_operation"], metadata["surface_name"],
            metadata["relation_orientation"] or "direct", metadata["distractor"], config,
        )
        round_trip += expected == row["agent_input"]["evidence_text"] and length == metadata["sentence_length_stratum"]
        compiler_correct += compile_truth(factor["lexical_sign"], factor["outer_operation"], config) == target["truth_status"]
        canonical += target["atom"] == atom_key(target["predicate"], target["arguments"], config)
        entity_types = {entity["id"]: entity["entity_type"] for entity in row["agent_input"]["entities"]}
        spec = specs[target["predicate"]]
        if spec["kind"] == "unary":
            valid = len(target["arguments"]) == 1 and entity_types.get(target["arguments"][0]) == spec["entityType"]
        else:
            valid = len(target["arguments"]) == 2 and entity_types.get(target["arguments"][0]) == spec["sourceType"] and entity_types.get(target["arguments"][1]) == spec["targetType"] and target["arguments"][0] != target["arguments"][1]
        type_valid += valid
        scene_atoms[row["scene_id"]].append(target["atom"])
        if metadata["construction_hash"] != construction_hash(factor["outer_operation"], metadata["surface_name"]):
            errors.append(f"Construction hash mismatch: {row['id']}")
    if leaks: errors.append(f"V32 target fields leaked into agent input: {dict(leaks)}")
    if len({row["id"] for row in rows}) != len(rows): errors.append("V32 record IDs are not unique")
    if any(len(values) != len(set(values)) for values in scene_atoms.values()): errors.append("V32 scene repeats an atom")
    if round_trip != len(rows): errors.append("V32 evidence round trip failed")
    if compiler_correct / len(rows) != gates["requiredOracleCompilerAccuracy"]: errors.append("V32 compiler oracle accuracy failed")
    if canonical != len(rows) or type_valid != len(rows): errors.append("V32 canonical/type audit failed")

    exact_overlap = construction_overlap = prompt_overlap = 0
    for index, left in enumerate(splits):
        for right in splits[index + 1:]:
            exact_overlap += len(exact_sets[left] & exact_sets[right])
            prompt_overlap += len(prompt_sets[left] & prompt_sets[right])
            construction_overlap += len(construction_sets[left] & construction_sets[right])
    old_overlap = len(set().union(*construction_sets.values()) & historical_hashes())
    if exact_overlap > gates["maximumExactEvidenceOverlapAcrossSplits"]: errors.append("V32 exact evidence overlaps splits")
    if construction_overlap > gates["maximumConstructionHashOverlapAcrossSplits"]: errors.append("V32 construction hashes overlap splits")
    if old_overlap > gates["maximumConstructionHashOverlapWithV30OrV31"]: errors.append("V32 construction hashes overlap V30/V31")

    fit_cells = {(row["target"]["factorization"]["outer_operation"], row["target"]["factorization"]["lexical_sign"]) for row in by_split["factor_fit"]}
    calibration_cells = {(row["target"]["factorization"]["outer_operation"], row["target"]["factorization"]["lexical_sign"]) for row in by_split["factor_calibration"]}
    paraphrase_cells = {(row["target"]["factorization"]["outer_operation"], row["target"]["factorization"]["lexical_sign"]) for row in by_split["factor_evaluation_paraphrase"]}
    composition_cells = {(row["target"]["factorization"]["outer_operation"], row["target"]["factorization"]["lexical_sign"]) for row in by_split["factor_evaluation_composition"]}
    expected_fit = {(operation, sign) for operation, signs in config["factorization"]["fitCells"].items() for sign in signs}
    expected_composition = {(operation, sign) for operation, signs in config["factorization"]["compositionHoldoutCells"].items() for sign in signs}
    if fit_cells != expected_fit or calibration_cells != expected_fit or paraphrase_cells != expected_fit:
        errors.append("V32 supported-cell populations differ from registration")
    if composition_cells != expected_composition:
        errors.append("V32 composition cells differ from registration")
    novel_composition = composition_cells - fit_cells
    expected_novel = expected_composition - expected_fit
    if novel_composition != expected_novel or len(expected_novel) != 3:
        errors.append("V32 composition novelty audit failed")
    fit_signs = {sign for _, sign in fit_cells}
    fit_operations = {operation for operation, _ in fit_cells}
    if fit_signs != set(config["factorization"]["lexicalSigns"]) or fit_operations != set(config["factorization"]["outerOperations"]):
        errors.append("V32 fit does not support all component values")
    para_families = {row["oracle_metadata"]["surface_family"] for row in by_split["factor_evaluation_paraphrase"]}
    comp_families = {row["oracle_metadata"]["surface_family"] for row in by_split["factor_evaluation_composition"]}
    if len(para_families) != gates["requiredEvaluationParaphraseFamilies"]: errors.append("Wrong V32 paraphrase family count")
    if len(comp_families) != gates["requiredEvaluationCompositionFamilies"]: errors.append("Wrong V32 composition family count")

    pair_groups = defaultdict(list)
    for row in rows:
        for pair in row["oracle_metadata"]["pairs"]:
            pair_groups[(pair["kind"], pair["id"])].append((row, pair))
    pair_errors, pair_counts = [], Counter()
    for (kind, identifier), members in pair_groups.items():
        pair_counts[kind] += 1
        if len(members) != 2:
            pair_errors.append(f"{kind}:{identifier} has {len(members)} members")
            continue
        (left, lp), (right, rp) = members
        if lp["role"] == rp["role"]: pair_errors.append(f"{kind}:{identifier} duplicates roles")
        lt, rt = left["target"], right["target"]
        if kind in ("distractor", "inverse", "unresolved_sign_invariance", "scope_assert_double_deny", "scope_assert_contrast"):
            if (lt["predicate"], lt["arguments"], lt["truth_status"]) != (rt["predicate"], rt["arguments"], rt["truth_status"]):
                pair_errors.append(f"{kind}:{identifier} should preserve signed fact")
        elif kind == "argument_reversal":
            if not (lt["predicate"] == rt["predicate"] == "linked" and lt["arguments"] == list(reversed(rt["arguments"])) and lt["truth_status"] == rt["truth_status"]):
                pair_errors.append(f"argument_reversal:{identifier} malformed")
        elif kind in ("lexical_sign_assert", "scope_assert_deny"):
            if not (lt["predicate"] == rt["predicate"] and lt["arguments"] == rt["arguments"] and {lt["truth_status"], rt["truth_status"]} == {"true", "false"}):
                pair_errors.append(f"{kind}:{identifier} should invert truth")
    if pair_errors: errors.extend(pair_errors[:10])
    required_pair_kinds = {
        "distractor", "inverse", "argument_reversal", "lexical_sign_assert",
        "unresolved_sign_invariance", "scope_assert_deny", "scope_assert_double_deny",
        "scope_assert_contrast",
    }
    if set(pair_counts) != required_pair_kinds: errors.append("V32 controlled pair inventory incomplete")

    expected = build_records(config)
    if [canonical_json(row) for row in sorted(rows, key=lambda row: row["id"])] != [canonical_json(row) for row in sorted(expected, key=lambda row: row["id"])]:
        errors.append("V32 on-disk corpus differs from reconstruction")
    root = PROJECT_ROOT / config["outputDir"]
    if manifest["config_sha256"] != file_sha256(config_path) or manifest["corpus_sha256"] != corpus_hash(list(rows)):
        errors.append("V32 manifest/config/corpus integrity mismatch")
    for name, expected_hash in manifest["artifact_sha256"].items():
        if file_sha256(root / name) != expected_hash: errors.append(f"V32 artifact changed: {name}")

    v31_result = json.loads((PROJECT_ROOT / config["sourceV31Result"]).read_text())
    v31_audit = json.loads((PROJECT_ROOT / config["sourceV31PostAudit"]).read_text())
    forensic = json.loads((PROJECT_ROOT / config["sourceV31ForensicAudit"]).read_text())
    if not (v31_audit["passed"] and not v31_result["passed"] and forensic["passed"]):
        errors.append("Accepted V31 evidence does not authorize V32")
    forbidden = (
        PROJECT_ROOT / "outputs/v32-factorized-semantics/fit-calibration-features",
        PROJECT_ROOT / "outputs/v32-factorized-semantics/training",
        PROJECT_ROOT / "outputs/v32-factorized-semantics/sealed-evaluation",
        PROJECT_ROOT / "configs/v32-trained-systems-lock.json",
    )
    if enforce_firewall and any(path.exists() for path in forbidden):
        errors.append("V32 model/training/evaluation artifact exists before protocol lock")
    return {
        "schema_version": 32, "experiment": "v32_pre_model_structural_audit",
        "passed": not errors,
        "decision": "authorize_v32_protocol_lock" if not errors else "repair_v32_before_model_access",
        "errors": errors,
        "population": {
            "records": len(rows), "scenes": len(scene_atoms),
            "split_counts": dict(sorted(Counter(row["split"] for row in rows).items())),
            "paraphrase_families": len(para_families), "composition_families": len(comp_families),
        },
        "semantics": {
            "oracle_compiler_accuracy": compiler_correct / len(rows),
            "oracle_round_trip_accuracy": round_trip / len(rows),
            "canonical_target_accuracy": canonical / len(rows),
            "type_validity_accuracy": type_valid / len(rows),
            "fit_cells": sorted(f"{operation}/{sign}" for operation, sign in fit_cells),
            "composition_cells": sorted(f"{operation}/{sign}" for operation, sign in composition_cells),
            "novel_composition_cells": sorted(f"{operation}/{sign}" for operation, sign in novel_composition),
            "pair_counts": dict(sorted(pair_counts.items())), "pair_errors": len(pair_errors),
        },
        "separation": {
            "exact_evidence_overlap_across_splits": exact_overlap,
            "exact_prompt_overlap_across_splits": prompt_overlap,
            "construction_hash_overlap_across_splits": construction_overlap,
            "construction_hash_overlap_with_v30_v31": old_overlap,
        },
        "firewall": {
            "target_fields_in_agent_input": dict(sorted(leaks.items())),
            "model_forward_passes_before_lock": 0, "training_runs_before_lock": 0,
            "evaluation_features_before_trained_lock": 0,
            "evaluation_predictions_before_trained_lock": 0,
        },
        "integrity": {
            "config_sha256": file_sha256(config_path),
            "manifest_sha256": file_sha256(root / "manifest.json") if (root / "manifest.json").exists() else None,
            "corpus_sha256": corpus_hash(list(rows)),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/v32-factorized-semantics.json")
    parser.add_argument("--output", default="outputs/v32-factorized-semantics/pre-model-audit.json")
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


if __name__ == "__main__":
    main()
