#!/usr/bin/env python3
"""Audit V43 implementation and dry-build the paired corpus without scoring it."""

from __future__ import annotations

import argparse
import json

from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT
from generate_v43_language import build_population, corpus_hash, population_counts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--design-lock", default="configs/v43-design-lock.json")
    parser.add_argument("--output", default="outputs/v43-sequential-language-grounding/implementation-audit.json")
    args = parser.parse_args()
    design_path = (PROJECT_ROOT / args.design_lock).resolve()
    output = (PROJECT_ROOT / args.output).resolve()
    design = json.loads(design_path.read_text())
    errors = []
    if not design["authorization"]["write_and_audit_implementation"]:
        errors.append("V43 design does not authorize implementation")
    source_seal_path = PROJECT_ROOT / design["source_v42_corpus_seal"]
    if file_sha256(source_seal_path) != design["source_v42_corpus_seal_sha256"]:
        errors.append("V42 source seal changed after V43 design lock")
    source_seal = json.loads(source_seal_path.read_text())
    v42_implementation_path = PROJECT_ROOT / source_seal["implementation_lock"]
    v42_implementation = json.loads(v42_implementation_path.read_text())
    if file_sha256(PROJECT_ROOT / "python/v42_stateful.py") != v42_implementation["implementation"]["python/v42_stateful.py"]:
        errors.append("Frozen V42 stateful reasoner changed")
    v39_lock_path = PROJECT_ROOT / "configs/v39-implementation-lock.json"
    v39_lock = json.loads(v39_lock_path.read_text())
    for path in ("python/v39_compiler.py", "python/v38_focus_parser.py"):
        if file_sha256(PROJECT_ROOT / path) != v39_lock["implementation"][path]:
            errors.append(f"Frozen language dependency changed: {path}")
    rows = build_population(source_seal_path)
    counts = population_counts(rows)
    expected = design["config_payload"]["pairedDesign"]
    if (counts["mechanics"], counts["support_sequences"], counts["query_sequences"]) != (expected["mechanics"], expected["supportSequences"], expected["querySequences"]):
        errors.append("V43 dry-build counts differ from preregistration")
    if counts["safety_challenges"] != 40 * len(design["config_payload"]["safetyChallenges"]):
        errors.append("V43 safety challenge quota failed")
    exposed_raw = any(
        any(key in sequence for key in ("initial_state", "actions", "observed_step_states"))
        for row in rows
        for section in (row["agent_input"]["support_sequences"], row["agent_input"]["queries"])
        for sequence in section
    )
    if exposed_raw:
        errors.append("Raw symbolic state or action sequence is exposed in V43 agent input")
    downstream = (
        "configs/v43-implementation-lock.json",
        "configs/v43-corpus-seal.json",
        "data/v43-sequential-language-grounding",
        "outputs/v43-sequential-language-grounding/development",
    )
    if any((PROJECT_ROOT / path).exists() for path in downstream):
        errors.append("V43 downstream artifact exists before implementation lock")
    dry_run = {
        **counts,
        "expected_corpus_sha256": corpus_hash(rows),
        "raw_symbolic_agent_input_fields": int(exposed_raw),
        "paired_development_predictions": 0,
    }
    audit = {
        "schema_version": 43,
        "experiment": "v43_implementation_audit",
        "passed": not errors,
        "decision": "authorize_v43_implementation_lock" if not errors else "repair_v43_implementation",
        "errors": errors,
        "design_lock": str(design_path.relative_to(PROJECT_ROOT)),
        "design_lock_sha256": file_sha256(design_path),
        "dry_run": dry_run,
        "data_access": {
            "v42_records_read": 40,
            "paired_development_runs": 0,
            "language_model_forward_passes": 0,
            "adapter_training_runs": 0,
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    print(json.dumps(audit, indent=2, sort_keys=True))
    if not audit["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
