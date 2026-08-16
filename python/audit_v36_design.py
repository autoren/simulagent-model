#!/usr/bin/env python3
"""Audit source authorization and internal consistency of the V36 design only."""

from __future__ import annotations

import argparse
import json

from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT


def selected_alpha(result: dict, collection: str, method: str) -> float:
    return float(result[collection][method]["selected_cv"]["alpha"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/v36-independent-confirmation-design.json")
    parser.add_argument("--output", default="outputs/v36-independent-confirmation/design-audit.json")
    args = parser.parse_args()
    config_path, output_path = (PROJECT_ROOT / args.config).resolve(), (PROJECT_ROOT / args.output).resolve()
    config, errors = json.loads(config_path.read_text()), []
    paths = {
        "v32_protocol": PROJECT_ROOT / config["sourceV32ProtocolLock"],
        "v32_features": PROJECT_ROOT / config["sourceV32FeatureMetadata"],
        "v34_protocol": PROJECT_ROOT / config["sourceV34ProtocolLock"],
        "v34_result": PROJECT_ROOT / config["sourceV34Result"],
        "v34_outcome": PROJECT_ROOT / config["sourceV34OutcomeLock"],
        "v35_protocol": PROJECT_ROOT / config["sourceV35ProtocolLock"],
        "v35_result": PROJECT_ROOT / config["sourceV35Result"],
        "v35_outcome": PROJECT_ROOT / config["sourceV35OutcomeLock"],
    }
    v34_result, v34_outcome, v35_result, v35_outcome = (
        json.loads(paths[name].read_text()) for name in ("v34_result", "v34_outcome", "v35_result", "v35_outcome")
    )
    if not v34_outcome["qualification"]["passed"] or not v35_outcome["qualification"]["passed"]:
        errors.append("Audited V34/V35 qualification is unavailable")
    if not v35_outcome["authorization"]["independent_confirmation_preregistration"]:
        errors.append("V35 outcome does not authorize V36 preregistration")
    if v35_outcome["authorization"]["independent_confirmation_construction"]:
        errors.append("V35 unexpectedly authorized construction before V36 design")
    frozen = config["frozenInterface"]
    expected = {
        "predicate": selected_alpha(v35_result, "predicate_methods", "atomHiddenRidge"),
        "binding": selected_alpha(v35_result, "binding_methods", "atomEvidenceEntityRoleRidge"),
        "lexicalSign": float(v35_result["lexical_sign_method"]["selected_cv"]["alpha"]),
        "outerOperation": float(v34_result["methods"]["semanticHiddenRidge"]["selected_cv"]["alpha"]),
    }
    for component, alpha in expected.items():
        if float(frozen[component]["alpha"]) != alpha:
            errors.append(f"V36 frozen {component} alpha differs from audited selection")
    suite = config["confirmationSuite"]
    families = len(suite["outerOperations"]) * len(suite["newSurfaceNamesPerOperation"])
    scenes = families * len(suite["lexicalSignsPerOperation"]) * suite["baseScenesPerSurfaceFamilyCell"] * len(suite["sceneVariants"])
    clauses_per_cell_base = 5 + 2 + 5 + 1
    records = families * len(suite["lexicalSignsPerOperation"]) * suite["baseScenesPerSurfaceFamilyCell"] * clauses_per_cell_base
    if families != suite["requiredSurfaceFamilies"] or scenes != suite["expectedScenes"] or records != suite["expectedRecords"]:
        errors.append("V36 confirmation population arithmetic mismatch")
    if config["execution"]["expectedBackboneForwardPasses"] != records * config["execution"]["featureExtractions"]:
        errors.append("V36 forward-pass budget mismatch")
    authorization = config["designStageAuthorization"]
    if not authorization["writeImplementation"] or any(authorization[key] for key in (
        "fitInterface", "constructConfirmation", "accessConfirmationLabels", "reuseV32Evaluation", "runV28", "constructFreshFinalSuite"
    )):
        errors.append("V36 design-stage authorization is too broad or incomplete")
    forbidden = (
        PROJECT_ROOT / "configs/v36-independent-confirmation-design-lock.json",
        PROJECT_ROOT / suite["outputDir"],
        PROJECT_ROOT / "outputs/v36-independent-confirmation/interface",
        PROJECT_ROOT / "outputs/v36-independent-confirmation/evaluation",
    )
    if any(path.exists() for path in forbidden):
        errors.append("V36 construction/model artifact exists before design lock")
    result = {
        "schema_version": 36, "experiment": "v36_design_audit", "passed": not errors,
        "decision": "authorize_v36_design_lock" if not errors else "repair_v36_design", "errors": errors,
        "derived_population": {"surface_families": families, "scenes": scenes, "records": records, "backbone_forward_passes": records * config["execution"]["featureExtractions"]},
        "frozen_component_alphas": expected,
        "source": {"config_sha256": file_sha256(config_path), **{f"{name}_sha256": file_sha256(path) for name, path in paths.items()}},
        "authorization": authorization,
        "data_access": {"confirmation_records_constructed": 0, "confirmation_labels_read": 0, "model_forward_passes": 0, "interface_fit_runs": 0, "v32_evaluation_records_read": 0, "v28_integration_replays": 0},
    }
    output_path.parent.mkdir(parents=True, exist_ok=True); output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
