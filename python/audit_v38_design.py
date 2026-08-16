#!/usr/bin/env python3
"""Audit and authorize freezing of the V38 parser-pivot design only."""

from __future__ import annotations

import argparse
import json

from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/v38-ontology-anchored-focus-parser.json")
    parser.add_argument("--output", default="outputs/v38-ontology-anchored-focus-parser/design-audit.json")
    args = parser.parse_args()
    config_path = (PROJECT_ROOT / args.config).resolve()
    output_path = (PROJECT_ROOT / args.output).resolve()
    config = json.loads(config_path.read_text())
    outcome_path = PROJECT_ROOT / config["sourceV37OutcomeLock"]
    diagnostic_path = PROJECT_ROOT / config["sourceV37Diagnostic"]
    outcome = json.loads(outcome_path.read_text())
    diagnostic = json.loads(diagnostic_path.read_text())
    errors = []
    if outcome["scientific_decision"] != "semantic_invariance_no_material_gain_pivot_parser_or_grounder":
        errors.append("V37 does not authorize the parser/grounder pivot")
    if outcome["authorization"]["change_backbone"]:
        errors.append("V37 unexpectedly authorizes an immediate backbone change")
    if any(
        diagnostic["corpora"][name]["lexical_sign_accuracy"] != 1.0
        for name in diagnostic["corpora"]
    ):
        errors.append("The lexical-anchor diagnostic does not support V38")
    if diagnostic["status"] != "descriptive_oracle_lexicon_not_deployable":
        errors.append("V38 fails to preserve the diagnostic limitation")
    stage = config["stageAuthorization"]
    if not stage["writeAndAuditImplementation"] or any(
        stage[key] for key in ("constructCorpus", "accessModel", "fitParser", "scoreValidation", "preregisterConfirmation")
    ):
        errors.append("V38 design-stage authorization is too broad")
    firewall = config["firewall"]
    for key in (
        "v32CalibrationOrEvaluationUse", "v28Use", "adapterTraining", "backboneChange", "endToEndRelationalSuite"
    ):
        if firewall[key] != "forbidden":
            errors.append(f"V38 firewall does not forbid {key}")
    forbidden = (
        PROJECT_ROOT / "configs/v38-ontology-anchored-focus-parser-lock.json",
        PROJECT_ROOT / "configs/v38-implementation-lock.json",
        PROJECT_ROOT / "data/v38-ontology-anchored-focus-parser",
        PROJECT_ROOT / "outputs/v38-ontology-anchored-focus-parser/features",
        PROJECT_ROOT / "outputs/v38-ontology-anchored-focus-parser/evaluation",
    )
    if any(path.exists() for path in forbidden):
        errors.append("V38 artifact exists before design lock")
    audit = {
        "schema_version": 38,
        "experiment": "v38_design_audit",
        "passed": not errors,
        "decision": "authorize_v38_design_lock" if not errors else "repair_v38_design",
        "errors": errors,
        "source": {
            "config_sha256": file_sha256(config_path),
            "v37_outcome_sha256": file_sha256(outcome_path),
            "v37_diagnostic_sha256": file_sha256(diagnostic_path),
            "plan_sha256": file_sha256(PROJECT_ROOT / "docs/v38-ontology-anchored-focus-parser-plan.md"),
        },
        "data_access": {
            "v38_records_constructed": 0,
            "model_forward_passes": 0,
            "fit_runs": 0,
            "validation_evaluations": 0,
            "v32_calibration_or_evaluation_records_read": 0,
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
