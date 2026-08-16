#!/usr/bin/env python3
"""Audit the complete V36 implementation before any readout fitting."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re

from generate_v36_confirmation import build_records, corpus_hash
from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT
from v32_language import SURFACE_TEMPLATES as V32_TEMPLATES
from v32_language import representation_prompt_layout
from v34_operation import operation_prompt
from v35_binding import atom_prompt_layout
from v36_language import (
    COLLISION_POLICY, GENERATOR_SEED, NORMALIZATION_VERSION, SURFACE_TEMPLATES,
    normalized_template, validate_registry,
)


IMPLEMENTATION = (
    "python/v36_language.py", "python/generate_v36_confirmation.py",
    "python/v36_interface.py", "python/fit_v36_interface.py",
    "python/audit_v36_interface.py", "python/freeze_v36_interface.py",
    "python/audit_v36_confirmation.py", "python/seal_v36_confirmation.py",
    "python/extract_v36_features_mlx.py", "python/freeze_v36_features.py",
    "python/v36_evaluation.py", "python/evaluate_v36_confirmation.py",
    "python/audit_and_summarize_v36.py", "python/freeze_v36_outcome.py",
    "python/audit_v36_implementation.py", "python/freeze_v36_implementation.py",
    "python/test_v36_language.py", "python/test_v36_interface.py", "python/test_v36_evaluation.py",
    "python/v34_operation.py", "python/v35_binding.py",
    "python/v32_language.py", "python/v30_language.py",
    "python/audit_v32_factorized_semantics.py", "python/extract_v10_features_mlx.py",
    "python/extract_v22r2_relational_features_mlx.py", "python/v10_protocol.py",
    "python/v22r2_grounding.py",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--design-lock", default="configs/v36-independent-confirmation-design-lock.json")
    parser.add_argument("--output", default="outputs/v36-independent-confirmation/implementation-audit.json")
    args = parser.parse_args()
    design_path, output_path = (PROJECT_ROOT / args.design_lock).resolve(), (PROJECT_ROOT / args.output).resolve()
    design = json.loads(design_path.read_text()); config = design["config_payload"]; errors = []
    if not design["authorization"]["write_implementation"] or any(design["authorization"][key] for key in ("fit_interface", "construct_confirmation", "model_access")):
        errors.append("V36 design lock does not isolate implementation stage")
    v32_lock = json.loads((PROJECT_ROOT / config["sourceV32ProtocolLock"]).read_text())
    v34_lock = json.loads((PROJECT_ROOT / config["sourceV34ProtocolLock"]).read_text())
    v35_lock = json.loads((PROJECT_ROOT / config["sourceV35ProtocolLock"]).read_text())
    v32_config, v34_config, v35_config = v32_lock["config_payload"], v34_lock["config_payload"], v35_lock["config_payload"]
    validate_registry(config)
    rows = build_records(config, v32_config)
    if len(rows) != 1170 or len({row["scene_id"] for row in rows}) != 360 or len({row["oracle_metadata"]["surface_family"] for row in rows}) != 15:
        errors.append("V36 in-memory generator population differs from design")
    old_normalized = {re.sub(r"\s+", " ", re.sub(r"\{[^}]+\}", "{SLOT}", template.lower())).strip() for values in V32_TEMPLATES.values() for template, _ in values.values()}
    new_normalized = {normalized_template(operation, surface) for operation, values in SURFACE_TEMPLATES.items() for surface in values}
    if old_normalized & new_normalized:
        errors.append("V36 surface registry overlaps V32 normalized constructions")
    prompt_dependencies = []
    runtime_v35 = {**v35_config, "v32_config": v32_config}
    for row in rows:
        mutated = copy.deepcopy(row); mutated["target"] = {"sentinel": "must not affect prompt"}
        if representation_prompt_layout(row, v32_config) != representation_prompt_layout(mutated, v32_config):
            prompt_dependencies.append(f"generic:{row['id']}")
        if operation_prompt(row, v34_config) != operation_prompt(mutated, v34_config):
            prompt_dependencies.append(f"operation:{row['id']}")
        if atom_prompt_layout(row, runtime_v35) != atom_prompt_layout(mutated, runtime_v35):
            prompt_dependencies.append(f"atom:{row['id']}")
    if prompt_dependencies:
        errors.append("V36 focused prompt depends on target")
    missing = [path for path in IMPLEMENTATION if not (PROJECT_ROOT / path).is_file()]
    if missing:
        errors.append(f"V36 implementation files missing: {missing}")
    training_paths = {
        "v32_features": "outputs/v32-factorized-semantics/fit-calibration-features/metadata.json",
        "v34_features": "outputs/v34-operation-interface/features/metadata.json",
        "v35_features": "outputs/v35-binding-assembly/features/metadata.json",
    }
    training_sources = {}
    for name, relative in training_paths.items():
        path = PROJECT_ROOT / relative
        metadata = json.loads(path.read_text()); artifact = PROJECT_ROOT / metadata["feature_artifact"]
        if file_sha256(artifact) != metadata["feature_artifact_sha256"]:
            errors.append(f"V36 training feature changed: {name}")
        training_sources[name] = {"path": relative, "sha256": file_sha256(path), "artifact": metadata["feature_artifact"], "artifact_sha256": file_sha256(artifact)}
    forbidden = (
        PROJECT_ROOT / "configs/v36-implementation-lock.json", PROJECT_ROOT / "configs/v36-interface-lock.json",
        PROJECT_ROOT / "configs/v36-confirmation-seal.json", PROJECT_ROOT / "configs/v36-features-lock.json",
        PROJECT_ROOT / "configs/v36-outcome-lock.json", PROJECT_ROOT / config["confirmationSuite"]["outputDir"],
        PROJECT_ROOT / "outputs/v36-independent-confirmation/interface-fit-attempt.json",
        PROJECT_ROOT / "outputs/v36-independent-confirmation/features", PROJECT_ROOT / "outputs/v36-independent-confirmation/evaluation",
    )
    if any(path.exists() for path in forbidden):
        errors.append("V36 post-implementation artifact exists before implementation lock")
    registry_payload = {
        "generator_seed": GENERATOR_SEED, "normalization_version": NORMALIZATION_VERSION,
        "collision_policy": COLLISION_POLICY, "surface_templates": SURFACE_TEMPLATES,
    }
    registry_hash = hashlib.sha256(json.dumps(registry_payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    result = {
        "schema_version": 36, "experiment": "v36_implementation_audit", "passed": not errors,
        "decision": "authorize_v36_implementation_lock" if not errors else "repair_v36_implementation", "errors": errors,
        "population_dry_run": {"records": len(rows), "scenes": len({row["scene_id"] for row in rows}), "surface_families": len({row["oracle_metadata"]["surface_family"] for row in rows}), "corpus_sha256": corpus_hash(rows)},
        "registry": registry_payload, "registry_sha256": registry_hash,
        "overlap": {"normalized_v32_construction_overlap": len(old_normalized & new_normalized)},
        "prompt_target_dependencies": len(prompt_dependencies), "implementation_files": list(IMPLEMENTATION),
        "training_sources": training_sources,
        "source": {"design_lock_sha256": file_sha256(design_path), "v32_protocol_lock_sha256": file_sha256(PROJECT_ROOT / config["sourceV32ProtocolLock"]), "v34_protocol_lock_sha256": file_sha256(PROJECT_ROOT / config["sourceV34ProtocolLock"]), "v35_protocol_lock_sha256": file_sha256(PROJECT_ROOT / config["sourceV35ProtocolLock"])},
        "data_access": {"in_memory_confirmation_records_for_structural_dry_run": len(rows), "confirmation_artifacts_written": 0, "interface_fit_runs": 0, "model_forward_passes": 0, "v32_evaluation_records_read": 0},
    }
    output_path.parent.mkdir(parents=True, exist_ok=True); output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
