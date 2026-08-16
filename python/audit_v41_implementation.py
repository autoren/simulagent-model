#!/usr/bin/env python3
"""Audit V41 construction without invoking the compiler or scoring confirmation targets."""

from __future__ import annotations

import argparse
from collections import Counter
import json

from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT
from generate_v41_confirmation import build_population, corpus_hash, old_program_keys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--design-lock", default="configs/v41-design-lock.json")
    parser.add_argument("--output", default="outputs/v41-relational-mechanic-confirmation/implementation-audit.json")
    args = parser.parse_args()
    design_path = (PROJECT_ROOT / args.design_lock).resolve()
    output = (PROJECT_ROOT / args.output).resolve()
    design = json.loads(design_path.read_text())
    config = design["config_payload"]
    errors = []
    compiler_path = PROJECT_ROOT / design["frozen_compiler"]
    kernel_path = PROJECT_ROOT / design["frozen_semantic_kernel"]
    if file_sha256(compiler_path) != design["frozen_compiler_sha256"]:
        errors.append("V39 compiler changed after V41 design lock")
    if file_sha256(kernel_path) != design["frozen_semantic_kernel_sha256"]:
        errors.append("V22 semantic kernel changed after V41 design lock")
    v22 = json.loads((PROJECT_ROOT / design["v22_config"]).read_text())
    v32_path = PROJECT_ROOT / "configs/v32-factorized-semantics.json"
    v32 = json.loads(v32_path.read_text())
    rows = build_population(config, v22, v32)
    old_keys = old_program_keys(config)
    new_keys = {row["target"]["program_key"] for row in rows}
    if len(rows) != 40 or len(new_keys) != 40 or old_keys & new_keys:
        errors.append("V41 population is not 40 unseen unique programs")
    family_counts = Counter(row["construction_family"] for row in rows)
    bit_family_counts = Counter((row["construction_family"], row["oracle_metadata"]["outcome_bits"]) for row in rows)
    if set(family_counts.values()) != {10} or len(family_counts) != 4:
        errors.append("V41 family balance failed")
    if any(bit_family_counts[(family, 1)] != 3 or bit_family_counts[(family, 2)] != 7 for family in config["population"]["families"]):
        errors.append("V41 outcome-bit quotas failed")
    axes = Counter(query["query_axis"] for row in rows for query in row["oracle_grounding"]["queries"])
    if set(axes) != set(config["population"]["queryAxes"]):
        errors.append("V41 query-axis coverage failed")
    support_counts = [len(row["agent_input"]["support_traces"]) for row in rows]
    if max(support_counts) > config["population"]["maximumSupportTraces"]:
        errors.append("V41 support budget exceeded")
    clause_count = sum(len(scene["evidence_packets"]) for row in rows for scene in row["agent_input"]["support_traces"] + row["agent_input"]["queries"])
    if any("target" in row["agent_input"] or "oracle_grounding" in row["agent_input"] or "language_reference" in row["agent_input"] for row in rows):
        errors.append("V41 oracle field appears inside agent input")
    forbidden = (
        PROJECT_ROOT / "configs/v41-implementation-lock.json",
        PROJECT_ROOT / "data/v41-relational-mechanic-confirmation",
        PROJECT_ROOT / "configs/v41-corpus-seal.json",
        PROJECT_ROOT / "outputs/v41-relational-mechanic-confirmation/evaluation",
    )
    if any(path.exists() for path in forbidden):
        errors.append("V41 downstream artifact exists before implementation lock")
    audit = {
        "schema_version": 41,
        "experiment": "v41_implementation_audit",
        "passed": not errors,
        "decision": "authorize_v41_implementation_lock" if not errors else "repair_v41_implementation",
        "errors": errors,
        "design_lock": str(design_path.relative_to(PROJECT_ROOT)),
        "design_lock_sha256": file_sha256(design_path),
        "frozen_compiler_sha256": file_sha256(compiler_path),
        "frozen_semantic_kernel_sha256": file_sha256(kernel_path),
        "v32_config_sha256": file_sha256(v32_path),
        "dry_run": {
            "mechanics": len(rows), "unique_target_programs": len(new_keys),
            "old_target_overlap": len(old_keys & new_keys), "family_counts": dict(sorted(family_counts.items())),
            "outcome_bit_family_counts": {f"{family}|{bits}": count for (family, bits), count in sorted(bit_family_counts.items())},
            "query_axis_counts": dict(sorted(axes.items())), "maximum_support_traces": max(support_counts),
            "support_scenes": sum(support_counts), "query_scenes": sum(len(row["agent_input"]["queries"]) for row in rows),
            "language_clauses": clause_count, "expected_corpus_sha256": corpus_hash(rows),
            "confirmation_clauses_compiled": 0, "confirmation_records_scored": 0,
        },
        "data_access": {"confirmation_scoring_runs": 0, "model_forward_passes": 0, "v22r2_evaluation_records_read": 0, "v28_runs": 0},
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    print(json.dumps(audit, indent=2, sort_keys=True))
    if not audit["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
