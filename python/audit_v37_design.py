#!/usr/bin/env python3
"""Audit the V37 development design before any corpus, feature, or fit exists."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/v37-semantic-invariance.json")
    parser.add_argument("--output", default="outputs/v37-semantic-invariance/design-audit.json")
    args = parser.parse_args()
    config_path = (PROJECT_ROOT / args.config).resolve()
    output_path = (PROJECT_ROOT / args.output).resolve()
    config = json.loads(config_path.read_text())
    errors: list[str] = []

    outcome_path = PROJECT_ROOT / config["sourceV36OutcomeLock"]
    result_path = PROJECT_ROOT / config["sourceV36Result"]
    localization_path = PROJECT_ROOT / config["sourceV36Localization"]
    outcome = json.loads(outcome_path.read_text())
    result = json.loads(result_path.read_text())
    if outcome["scientific_decision"] != "confirmation_fail_reopen_semantic_interface_only":
        errors.append("V36 does not authorize semantic-interface-only development")
    if result["decision"] != outcome["scientific_decision"]:
        errors.append("V36 result and frozen outcome disagree")
    if result["metrics"]["atom_exact_accuracy"] != 1.0:
        errors.append("V36 did not isolate a semantic-only failure")

    fit = config["developmentFit"]
    if fit["recordsPerOperationSignCell"] * 10 != fit["expectedRecords"]:
        errors.append("V37 fit population arithmetic mismatch")
    validation = config["freshValidation"]
    families = 5 * len(validation["surfaceNamesPerOperation"])
    per_cell = 5 + 2 + 5 + 5 + 1
    records = families * len(validation["lexicalSignsPerOperation"]) * validation["baseScenesPerSurfaceFamilyCell"] * per_cell
    scenes = families * len(validation["lexicalSignsPerOperation"]) * validation["baseScenesPerSurfaceFamilyCell"] * len(validation["sceneVariants"])
    if (families, records, scenes) != (
        validation["expectedSurfaceFamilies"], validation["expectedRecords"], validation["expectedScenes"]
    ):
        errors.append("V37 validation population arithmetic mismatch")
    total = fit["expectedRecords"] + validation["expectedRecords"]
    if total * config["interfaces"]["viewsPerRecord"] != config["execution"]["expectedBackboneForwardPasses"]:
        errors.append("V37 forward budget mismatch")

    firewall = config["firewall"]
    required_forbidden = (
        "v32EvaluationUse", "v28Use", "atomInterfaceChanges", "executorChanges",
        "adapterOrBackboneTraining", "endToEndRelationalSuite",
    )
    if any(firewall[key] != "forbidden" for key in required_forbidden):
        errors.append("V37 firewall is incomplete")
    if firewall["freshValidationMaySelectMethodOrAlpha"]:
        errors.append("V37 validation may not select a method or alpha")

    forbidden_artifacts = (
        PROJECT_ROOT / "configs/v37-semantic-invariance-lock.json",
        PROJECT_ROOT / "configs/v37-implementation-lock.json",
        PROJECT_ROOT / "configs/v37-corpus-seal.json",
        PROJECT_ROOT / "data/v37-semantic-invariance",
        PROJECT_ROOT / "outputs/v37-semantic-invariance/features",
        PROJECT_ROOT / "outputs/v37-semantic-invariance/evaluation",
    )
    if any(path.exists() for path in forbidden_artifacts):
        errors.append("V37 corpus/model artifact exists before design lock")

    sources = {
        "v32_protocol": PROJECT_ROOT / config["sourceV32ProtocolLock"],
        "v36_outcome": outcome_path,
        "v36_result": result_path,
        "v36_localization": localization_path,
        "plan": PROJECT_ROOT / "docs/v37-semantic-invariance-plan.md",
    }
    audit = {
        "schema_version": 37,
        "experiment": "v37_design_audit",
        "passed": not errors,
        "decision": "authorize_v37_design_lock" if not errors else "repair_v37_design",
        "errors": errors,
        "derived_population": {
            "fit_records": fit["expectedRecords"],
            "validation_records": records,
            "validation_scenes": scenes,
            "surface_families": families,
            "backbone_forward_passes": total * config["interfaces"]["viewsPerRecord"],
        },
        "source": {
            "config_sha256": file_sha256(config_path),
            **{f"{name}_sha256": file_sha256(path) for name, path in sources.items()},
        },
        "data_access": {
            "fit_labels_read": 0,
            "validation_records_constructed": 0,
            "backbone_forward_passes": 0,
            "fit_runs": 0,
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
