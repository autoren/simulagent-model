#!/usr/bin/env python3
"""Structural and firewall audit for the materialized V37 corpora."""

from __future__ import annotations

from collections import Counter, defaultdict
import argparse
import json
import re

from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT
from v32_language import SURFACE_TEMPLATES as V32_TEMPLATES
from v36_language import SURFACE_TEMPLATES as V36_TEMPLATES
from generate_v37_semantic_invariance import corpus_hash, read_jsonl
from v37_language import normalized_template


def source_templates(registry):
    return {
        re.sub(r"\s+", " ", re.sub(r"\{[^}]+\}", "{SLOT}", template.lower())).strip()
        for values in registry.values() for template, _ in values.values()
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--implementation-lock", default="configs/v37-implementation-lock.json")
    parser.add_argument("--output", default="outputs/v37-semantic-invariance/corpus-audit.json")
    args = parser.parse_args()
    lock_path = (PROJECT_ROOT / args.implementation_lock).resolve()
    output_path = (PROJECT_ROOT / args.output).resolve()
    lock = json.loads(lock_path.read_text())
    config = lock["config_payload"]
    base = PROJECT_ROOT / "data/v37-semantic-invariance"
    fit_path = base / "semantic_invariance_fit.jsonl"
    validation_path = base / "semantic_invariance_validation.jsonl"
    fit_rows, validation_rows = read_jsonl(fit_path), read_jsonl(validation_path)
    errors: list[str] = []
    if len(fit_rows) != 400 or len(validation_rows) != 360:
        errors.append("V37 corpus counts differ from lock")
    if corpus_hash(fit_rows) != lock["expected_corpora"]["fit_corpus_sha256"]:
        errors.append("V37 fit corpus differs from dry-run lock")
    if corpus_hash(validation_rows) != lock["expected_corpora"]["validation_corpus_sha256"]:
        errors.append("V37 validation corpus differs from dry-run lock")
    if len({row["scene_id"] for row in validation_rows}) != 100:
        errors.append("V37 validation scene count differs from lock")
    if len({row["oracle_metadata"]["surface_family"] for row in validation_rows}) != 10:
        errors.append("V37 validation family count differs from lock")
    cell_counts = Counter(
        (row["target"]["factorization"]["outer_operation"], row["target"]["factorization"]["lexical_sign"])
        for row in fit_rows
    )
    if set(cell_counts.values()) != {40} or len(cell_counts) != 10:
        errors.append("V37 fit cells are not exactly balanced")
    source_counts = Counter(row["oracle_metadata"]["v37_source"] for row in fit_rows)
    if source_counts != {"v32_factor_fit": 112, "v36_exposed_confirmation": 288}:
        errors.append("V37 fit source quotas differ from lock")

    old = source_templates(V32_TEMPLATES) | source_templates(V36_TEMPLATES)
    new = {
        normalized_template(operation, surface)
        for operation in config["interfaces"]["outerOperationClasses"]
        for surface in config["freshValidation"]["surfaceNamesPerOperation"]
    }
    template_overlap = old & new
    source_evidence = set()
    for source in config["allowedTrainingSources"]:
        source_evidence.update(row["agent_input"]["evidence_text"] for row in read_jsonl(PROJECT_ROOT / source["corpus"]))
    evidence_overlap = source_evidence & {row["agent_input"]["evidence_text"] for row in validation_rows}
    if template_overlap or evidence_overlap:
        errors.append("V37 validation overlaps source language")
    if any("target" in row["agent_input"] for row in fit_rows + validation_rows):
        errors.append("V37 agent input exposes a target")

    pair_members: dict[tuple[str, str], list[str]] = defaultdict(list)
    for row in validation_rows:
        for pair in row["oracle_metadata"]["pairs"]:
            pair_members[(pair["kind"], pair["id"])].append(row["id"])
    pair_counts = Counter(kind for kind, _ in pair_members)
    for (kind, _), identifiers in pair_members.items():
        expected = 3 if kind == "distractor_position" else 2
        if len(identifiers) != expected:
            errors.append(f"V37 malformed {kind} pair")

    audit = {
        "schema_version": 37,
        "experiment": "v37_corpus_audit",
        "passed": not errors,
        "decision": "authorize_v37_corpus_seal" if not errors else "repair_v37_corpus",
        "errors": sorted(set(errors)),
        "implementation_lock": str(lock_path.relative_to(PROJECT_ROOT)),
        "implementation_lock_sha256": file_sha256(lock_path),
        "artifacts": {
            "fit": {"path": str(fit_path.relative_to(PROJECT_ROOT)), "sha256": file_sha256(fit_path), "records": len(fit_rows)},
            "validation": {"path": str(validation_path.relative_to(PROJECT_ROOT)), "sha256": file_sha256(validation_path), "records": len(validation_rows)},
        },
        "overlap_checks": {
            "normalized_template_overlap": len(template_overlap),
            "exact_evidence_overlap": len(evidence_overlap),
        },
        "pair_counts": dict(sorted(pair_counts.items())),
        "data_access": {
            "backbone_forward_passes": 0,
            "fit_runs": 0,
            "validation_evaluations": 0,
            "v32_evaluation_records_read": 0,
            "v28_runs": 0,
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    print(json.dumps(audit, indent=2, sort_keys=True))
    if not audit["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
