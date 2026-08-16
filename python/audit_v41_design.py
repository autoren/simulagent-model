#!/usr/bin/env python3
"""Audit V41 preregistration and unseen-program capacity."""

from __future__ import annotations

import argparse
import json

from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT
from v22_relational import target_hypotheses


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/v41-relational-mechanic-confirmation.json")
    parser.add_argument("--output", default="outputs/v41-relational-mechanic-confirmation/design-audit.json")
    args = parser.parse_args()
    config_path = (PROJECT_ROOT / args.config).resolve()
    output = (PROJECT_ROOT / args.output).resolve()
    config = json.loads(config_path.read_text())
    errors = []
    source_path = PROJECT_ROOT / config["sourceV40OutcomeLock"]
    source = json.loads(source_path.read_text()) if source_path.is_file() else {}
    if not source.get("qualification_passed") or not source.get("authorization", {}).get("preregister_relational_mechanic_confirmation"):
        errors.append("V40 does not authorize V41 preregistration")
    corpus_root = PROJECT_ROOT / config["sourceV22DevelopmentCorpus"] / "records"
    old_records = [row for path in sorted(corpus_root.glob("*.jsonl")) for row in read_jsonl(path)]
    old_keys = {row["target"]["program_key"] for row in old_records}
    capacity = {}
    for family in config["population"]["families"]:
        capacity[family] = {}
        for bits, needed in ((1, config["population"]["oneBitMechanicsPerFamily"]), (2, config["population"]["twoBitMechanicsPerFamily"])):
            available = [row for row in target_hypotheses(family, bits) if row.key not in old_keys]
            capacity[family][str(bits)] = len(available)
            if len(available) < needed:
                errors.append(f"Insufficient unseen programs for {family}/{bits}")
    population = config["population"]
    if population["mechanics"] != 40 or population["mechanicsPerFamily"] != 10:
        errors.append("V41 mechanic population is not fixed at 40")
    if population["oneBitMechanicsPerFamily"] + population["twoBitMechanicsPerFamily"] != population["mechanicsPerFamily"]:
        errors.append("V41 outcome-bit quotas do not sum to family size")
    if config["firewall"]["confirmationRepeats"] != 0 or config["firewall"]["compilerModification"] != "forbidden":
        errors.append("V41 confirmation firewall is incomplete")
    if any((PROJECT_ROOT / path).exists() for path in ("configs/v41-design-lock.json", "data/v41-relational-mechanic-confirmation", "outputs/v41-relational-mechanic-confirmation/evaluation")):
        errors.append("V41 downstream artifact exists before design lock")
    audit = {
        "schema_version": 41,
        "experiment": "v41_design_audit",
        "passed": not errors,
        "decision": "authorize_v41_design_lock" if not errors else "repair_v41_design",
        "errors": errors,
        "config": str(config_path.relative_to(PROJECT_ROOT)),
        "config_sha256": file_sha256(config_path),
        "source_v40_outcome_lock": str(source_path.relative_to(PROJECT_ROOT)),
        "source_v40_outcome_lock_sha256": file_sha256(source_path) if source_path.is_file() else None,
        "old_v22_target_programs": len(old_keys),
        "unseen_program_capacity": capacity,
        "checks": {
            "forty_mechanics": population["mechanics"] == 40,
            "four_balanced_families": len(population["families"]) == 4,
            "all_old_program_keys_excluded": population["excludeAllV22TargetProgramKeys"],
            "one_confirmation": config["firewall"]["confirmationRepeats"] == 0,
            "declared_scope_only": config["firewall"]["claimExpansionBeyondDeclaredOneStepBooleanScope"] == "forbidden",
        },
        "data_access": {"fresh_mechanics_constructed": 0, "confirmation_records_scored": 0, "v22r2_evaluation_records_read": 0, "model_forward_passes": 0, "v28_runs": 0},
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    print(json.dumps(audit, indent=2, sort_keys=True))
    if not audit["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
