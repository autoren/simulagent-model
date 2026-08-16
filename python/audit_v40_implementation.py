#!/usr/bin/env python3
"""Audit the independent V40 generator without invoking the frozen compiler."""

from __future__ import annotations

import argparse
from collections import Counter
import json

from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT
from generate_v40_confirmation import PACK_NAMES, build_populations, corpus_hash, ontology_pack


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--design-lock", default="configs/v40-design-lock.json")
    parser.add_argument("--output", default="outputs/v40-independent-compiler-confirmation/implementation-audit.json")
    args = parser.parse_args()
    design_path = (PROJECT_ROOT / args.design_lock).resolve()
    output = (PROJECT_ROOT / args.output).resolve()
    design = json.loads(design_path.read_text())
    config = design["config_payload"]
    errors = []
    compiler_path = PROJECT_ROOT / design["frozen_compiler"]
    if file_sha256(compiler_path) != design["frozen_compiler_sha256"]:
        errors.append("Frozen V39 compiler changed before V40 implementation")
    populations = build_populations(config)
    core = populations["independent_confirmation"]
    safety = populations["independent_safety"]
    if len(core) != 1440 or len(safety) != 120:
        errors.append("V40 population counts differ from preregistration")
    pack_counts = Counter(row["ontology_pack"] for row in core)
    if set(pack_counts) != set(PACK_NAMES) or set(pack_counts.values()) != {120}:
        errors.append("V40 ontology packs are not balanced")
    safety_counts = Counter(row["expected"]["condition"] for row in safety)
    if set(safety_counts) != set(config["safetyPopulation"]["conditions"]) or set(safety_counts.values()) != {24}:
        errors.append("V40 safety conditions are not balanced")
    operation_sign = Counter((row["oracle_metadata"]["operation"], row["oracle_metadata"]["sign"]) for row in core)
    if len(operation_sign) != 10 or set(operation_sign.values()) != {144}:
        errors.append("V40 operation/sign cells are not balanced")
    v32 = json.loads((PROJECT_ROOT / "configs/v32-factorized-semantics.json").read_text())
    prior_forms = {
        value for predicate in v32["ontology"]["relations"]
        for key, value in predicate.items() if key.endswith("Form")
    }
    fresh_forms = {
        value for index in range(len(PACK_NAMES)) for predicate in ontology_pack(index)[0]["relations"]
        for key, value in predicate.items() if key.endswith("_form")
    }
    if prior_forms & fresh_forms:
        errors.append("V40 relation lexical forms overlap V32")
    if any("target" in row["agent_input"] or "expected" in row["agent_input"] for rows in populations.values() for row in rows):
        errors.append("V40 target information appears in agent input")
    forbidden = (
        PROJECT_ROOT / "configs/v40-implementation-lock.json",
        PROJECT_ROOT / "data/v40-independent-compiler-confirmation",
        PROJECT_ROOT / "configs/v40-corpus-seal.json",
        PROJECT_ROOT / "outputs/v40-independent-compiler-confirmation/evaluation",
    )
    if any(path.exists() for path in forbidden):
        errors.append("V40 downstream artifact exists before implementation lock")
    audit = {
        "schema_version": 40,
        "experiment": "v40_implementation_audit",
        "passed": not errors,
        "decision": "authorize_v40_implementation_lock" if not errors else "repair_v40_implementation",
        "errors": errors,
        "design_lock": str(design_path.relative_to(PROJECT_ROOT)),
        "design_lock_sha256": file_sha256(design_path),
        "frozen_compiler": str(compiler_path.relative_to(PROJECT_ROOT)),
        "frozen_compiler_sha256": file_sha256(compiler_path),
        "dry_run": {
            "population_counts": {name: len(rows) for name, rows in populations.items()},
            "ontology_pack_counts": dict(sorted(pack_counts.items())),
            "safety_condition_counts": {str(key): value for key, value in sorted(safety_counts.items())},
            "operation_sign_cells": len(operation_sign),
            "fresh_relation_forms": len(fresh_forms),
            "expected_corpus_sha256": {name: corpus_hash(rows) for name, rows in populations.items()},
            "confirmation_records_scored": 0,
        },
        "data_access": {"confirmation_scoring_runs": 0, "model_forward_passes": 0, "v32_evaluation_records_read": 0, "v28_runs": 0},
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    print(json.dumps(audit, indent=2, sort_keys=True))
    if not audit["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
