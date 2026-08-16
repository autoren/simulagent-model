#!/usr/bin/env python3
"""Audit the V40 confirmation preregistration before implementation."""

from __future__ import annotations

import argparse
import json

from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/v40-independent-compiler-confirmation.json")
    parser.add_argument("--output", default="outputs/v40-independent-compiler-confirmation/design-audit.json")
    args = parser.parse_args()
    config_path = (PROJECT_ROOT / args.config).resolve()
    output = (PROJECT_ROOT / args.output).resolve()
    config = json.loads(config_path.read_text())
    errors = []
    source_path = PROJECT_ROOT / config["sourceV39OutcomeLock"]
    if not source_path.is_file():
        errors.append("V40 source V39 outcome lock is missing")
        source = {}
    else:
        source = json.loads(source_path.read_text())
    if not source.get("qualification_passed") or not source.get("authorization", {}).get("preregister_fresh_supported_language_confirmation"):
        errors.append("V39 does not authorize V40 preregistration")
    if config["confirmationPopulation"]["ontologyPacks"] != 12 or config["confirmationPopulation"]["coreRecordsPerPack"] != 120:
        errors.append("V40 population size is not fixed")
    if config["safetyPopulation"]["totalRecords"] != 120:
        errors.append("V40 safety population size is not fixed")
    if set(config["gates"].values()) != {1.0}:
        errors.append("V40 gates are not exact")
    firewall = config["firewall"]
    if firewall["compilerModificationAfterV39"] != "forbidden" or firewall["evaluationRepeats"] != 0:
        errors.append("V40 compiler or evaluation firewall is incomplete")
    if any((PROJECT_ROOT / path).exists() for path in ("configs/v40-design-lock.json", "data/v40-independent-compiler-confirmation", "outputs/v40-independent-compiler-confirmation/evaluation")):
        errors.append("V40 downstream artifact exists before design lock")
    audit = {
        "schema_version": 40,
        "experiment": "v40_design_audit",
        "passed": not errors,
        "decision": "authorize_v40_design_lock" if not errors else "repair_v40_design",
        "errors": errors,
        "config": str(config_path.relative_to(PROJECT_ROOT)),
        "config_sha256": file_sha256(config_path),
        "source_v39_outcome_lock": str(source_path.relative_to(PROJECT_ROOT)),
        "source_v39_outcome_lock_sha256": file_sha256(source_path) if source_path.is_file() else None,
        "checks": {
            "compiler_frozen_before_confirmation": True,
            "ontology_pack_reporting_unit": config["confirmationPopulation"]["statisticalUnit"] == "ontology_pack",
            "one_confirmation_evaluation": firewall["evaluationRepeats"] == 0,
            "open_paraphrase_excluded": firewall["claimExpansionToOpenParaphrase"] == "forbidden",
        },
        "data_access": {"confirmation_records_constructed": 0, "confirmation_records_scored": 0, "model_forward_passes": 0, "v32_evaluation_records_read": 0, "v28_runs": 0},
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    print(json.dumps(audit, indent=2, sort_keys=True))
    if not audit["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
