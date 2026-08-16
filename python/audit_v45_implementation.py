#!/usr/bin/env python3
"""Audit V45 implementation and dry-build the paired corpus without scoring it."""

from __future__ import annotations

import argparse
import json

from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT
from generate_v45_language import build_population, corpus_hash, population_counts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--design-lock", default="configs/v45-design-lock.json")
    parser.add_argument("--output", default="outputs/v45-delayed-language-grounding/implementation-audit.json")
    args = parser.parse_args()
    design_path = (PROJECT_ROOT / args.design_lock).resolve()
    output = (PROJECT_ROOT / args.output).resolve()
    design = json.loads(design_path.read_text())
    errors = []
    if not design["authorization"]["write_and_audit_implementation"]:
        errors.append("V45 design does not authorize implementation")
    source_seal_path = PROJECT_ROOT / design["source_v44_corpus_seal"]
    if file_sha256(source_seal_path) != design["source_v44_corpus_seal_sha256"]:
        errors.append("V44 source seal changed after V45 design lock")
    source_seal = json.loads(source_seal_path.read_text())
    v44_implementation_path = PROJECT_ROOT / source_seal["implementation_lock"]
    v44_implementation = json.loads(v44_implementation_path.read_text())
    if file_sha256(PROJECT_ROOT / "python/v44_delayed.py") != v44_implementation["implementation"]["python/v44_delayed.py"]:
        errors.append("Frozen V44 delayed reasoner changed")
    v39_lock = json.loads((PROJECT_ROOT / "configs/v39-implementation-lock.json").read_text())
    for path in ("python/v39_compiler.py", "python/v38_focus_parser.py"):
        if file_sha256(PROJECT_ROOT / path) != v39_lock["implementation"][path]:
            errors.append(f"Frozen language dependency changed: {path}")
    v43r1_lock = json.loads((PROJECT_ROOT / "configs/v43r1-implementation-lock.json").read_text())
    if file_sha256(PROJECT_ROOT / "python/v43r1_measurement.py") != v43r1_lock["implementation"]["python/v43r1_measurement.py"]:
        errors.append("Frozen canonical graph comparator changed")
    rows = build_population(source_seal_path)
    counts = population_counts(rows)
    paired = design["config_payload"]["pairedDesign"]
    if (counts["mechanics"], counts["support_sequences"], counts["query_sequences"], counts["wait_counterfactual_pairs"]) != (
        paired["mechanics"], paired["supportSequences"], paired["querySequences"], paired["waitCounterfactualPairs"],
    ):
        errors.append("V45 dry-build counts differ from preregistration")
    if counts["safety_challenges"] != 40 * len(design["config_payload"]["safetyChallenges"]):
        errors.append("V45 safety challenge quota failed")
    exposed_raw = any(
        any(key in sequence for key in ("initial_state", "actions", "observed_step_states"))
        for row in rows
        for section in (row["agent_input"]["support_sequences"], row["agent_input"]["queries"])
        for sequence in section
    )
    if exposed_raw:
        errors.append("Raw symbolic state or action sequence is exposed in V45 agent input")
    tick_mismatches = sum(
        row["agent_input"]["tick_semantics"] != v44_implementation["config_payload"]["tickSemantics"]
        for row in rows
    )
    if tick_mismatches:
        errors.append("V45 tick semantics differ from frozen V44")
    downstream = (
        "configs/v45-implementation-lock.json", "configs/v45-corpus-seal.json",
        "data/v45-delayed-language-grounding", "outputs/v45-delayed-language-grounding/development",
    )
    if any((PROJECT_ROOT / path).exists() for path in downstream):
        errors.append("V45 downstream artifact exists before implementation lock")
    dry_run = {
        **counts,
        "expected_corpus_sha256": corpus_hash(rows),
        "raw_symbolic_agent_input_fields": int(exposed_raw),
        "tick_semantic_mismatches": tick_mismatches,
        "paired_development_predictions": 0,
    }
    audit = {
        "schema_version": 45,
        "experiment": "v45_implementation_audit",
        "passed": not errors,
        "decision": "authorize_v45_implementation_lock" if not errors else "repair_v45_implementation",
        "errors": errors,
        "design_lock": str(design_path.relative_to(PROJECT_ROOT)),
        "design_lock_sha256": file_sha256(design_path),
        "dry_run": dry_run,
        "data_access": {"v44_records_read": 40, "paired_development_runs": 0, "model_forward_passes": 0, "adapter_training_runs": 0},
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    print(json.dumps(audit, indent=2, sort_keys=True))
    if not audit["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
