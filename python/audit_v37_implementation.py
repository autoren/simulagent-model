#!/usr/bin/env python3
"""Audit the complete V37 implementation in memory before corpus construction."""

from __future__ import annotations

import argparse
import copy
import json
import re

from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT
from v32_language import SURFACE_TEMPLATES as V32_TEMPLATES
from v36_language import SURFACE_TEMPLATES as V36_TEMPLATES
from generate_v37_semantic_invariance import build_fit_sample, build_validation, corpus_hash
from v37_language import candidate_prompt, normalized_template, validate_registry


def source_normalized_templates(registry):
    return {
        re.sub(r"\s+", " ", re.sub(r"\{[^}]+\}", "{SLOT}", template.lower())).strip()
        for values in registry.values() for template, _ in values.values()
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--design-lock", default="configs/v37-semantic-invariance-lock.json")
    parser.add_argument("--output", default="outputs/v37-semantic-invariance/implementation-audit.json")
    args = parser.parse_args()
    design_path = (PROJECT_ROOT / args.design_lock).resolve()
    output_path = (PROJECT_ROOT / args.output).resolve()
    design = json.loads(design_path.read_text())
    config = design["config_payload"]
    errors: list[str] = []
    if not design["authorization"]["write_implementation"]:
        errors.append("V37 design lock does not authorize implementation")
    if file_sha256(PROJECT_ROOT / design["config"]) != design["config_sha256"]:
        errors.append("V37 config changed after design lock")
    validate_registry(config)
    v32_config = json.loads((PROJECT_ROOT / "configs/v32-factorized-semantics.json").read_text())
    fit_rows = build_fit_sample(config, v32_config)
    validation_rows = build_validation(config, v32_config)
    if len(fit_rows) != config["developmentFit"]["expectedRecords"]:
        errors.append("V37 fit sample count mismatch")
    validation = config["freshValidation"]
    if len(validation_rows) != validation["expectedRecords"]:
        errors.append("V37 validation count mismatch")
    if len({row["scene_id"] for row in validation_rows}) != validation["expectedScenes"]:
        errors.append("V37 validation scene count mismatch")
    if len({row["oracle_metadata"]["surface_family"] for row in validation_rows}) != validation["expectedSurfaceFamilies"]:
        errors.append("V37 validation family count mismatch")

    old_templates = source_normalized_templates(V32_TEMPLATES) | source_normalized_templates(V36_TEMPLATES)
    new_templates = {
        normalized_template(operation, surface)
        for operation in config["interfaces"]["outerOperationClasses"]
        for surface in validation["surfaceNamesPerOperation"]
    }
    template_overlap = sorted(old_templates & new_templates)
    if template_overlap:
        errors.append("V37 normalized validation templates overlap V32/V36")
    source_evidence = set()
    for source in config["allowedTrainingSources"]:
        source_evidence.update(
            json.loads(line)["agent_input"]["evidence_text"]
            for line in (PROJECT_ROOT / source["corpus"]).read_text().splitlines() if line.strip()
        )
    validation_evidence = {row["agent_input"]["evidence_text"] for row in validation_rows}
    evidence_overlap = sorted(source_evidence & validation_evidence)
    if evidence_overlap:
        errors.append("V37 validation evidence overlaps allowed training sources")

    example = validation_rows[0]
    changed = copy.deepcopy(example)
    changed["target"] = {"sentinel": "must not be visible"}
    for component, candidates in (
        ("lexical_sign", config["interfaces"]["lexicalSignClasses"]),
        ("outer_operation", config["interfaces"]["outerOperationClasses"]),
    ):
        for candidate in candidates:
            if candidate_prompt(example, component, candidate) != candidate_prompt(changed, component, candidate):
                errors.append(f"V37 {component} prompt depends on target")

    forbidden = (
        PROJECT_ROOT / "configs/v37-implementation-lock.json",
        PROJECT_ROOT / "configs/v37-corpus-seal.json",
        PROJECT_ROOT / "data/v37-semantic-invariance",
        PROJECT_ROOT / "outputs/v37-semantic-invariance/features",
        PROJECT_ROOT / "outputs/v37-semantic-invariance/evaluation",
    )
    if any(path.exists() for path in forbidden):
        errors.append("V37 corpus/model artifact exists before implementation lock")

    audit = {
        "schema_version": 37,
        "experiment": "v37_implementation_audit",
        "passed": not errors,
        "decision": "authorize_v37_implementation_lock" if not errors else "repair_v37_implementation",
        "errors": errors,
        "design_lock": str(design_path.relative_to(PROJECT_ROOT)),
        "design_lock_sha256": file_sha256(design_path),
        "dry_run": {
            "fit_records": len(fit_rows),
            "validation_records": len(validation_rows),
            "validation_scenes": len({row["scene_id"] for row in validation_rows}),
            "validation_surface_families": len({row["oracle_metadata"]["surface_family"] for row in validation_rows}),
            "fit_corpus_sha256": corpus_hash(fit_rows),
            "validation_corpus_sha256": corpus_hash(validation_rows),
        },
        "overlap_checks": {
            "normalized_template_overlap": len(template_overlap),
            "exact_validation_evidence_overlap": len(evidence_overlap),
        },
        "data_access": {
            "fit_records_sampled_in_memory": len(fit_rows),
            "validation_records_built_in_memory": len(validation_rows),
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
