#!/usr/bin/env python3
"""Reproduce, audit, and summarize the V45 paired development result."""

from __future__ import annotations

import argparse
import json

from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT
from evaluate_v45_language import aggregate, evaluate_record, qualification, read
from v44_delayed import mechanic_registry


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", default="outputs/v45-delayed-language-grounding/development/result.json")
    parser.add_argument("--audit", default="outputs/v45-delayed-language-grounding/post-result-audit.json")
    parser.add_argument("--markdown", default="docs/v45-results.md")
    args = parser.parse_args()
    result_path, audit_path, markdown_path = tuple((PROJECT_ROOT / value).resolve() for value in (args.result, args.audit, args.markdown))
    result = json.loads(result_path.read_text())
    seal_path = PROJECT_ROOT / result["corpus_seal"]
    seal = json.loads(seal_path.read_text())
    implementation_path = PROJECT_ROOT / seal["implementation_lock"]
    implementation = json.loads(implementation_path.read_text())
    records = []
    for artifact in seal["corpora"].values():
        records.extend(read(PROJECT_ROOT / artifact["path"]))
    records.sort(key=lambda row: row["id"])
    registry = mechanic_registry()
    reproduced_metrics = aggregate([evaluate_record(row, registry)[0] for row in records])
    reproduced_qualification = qualification(reproduced_metrics, implementation["config_payload"]["gates"])
    reproduction = {
        "metrics": reproduced_metrics == result["metrics"],
        "qualification": reproduced_qualification == result["qualification"],
        "predictions_hash": file_sha256(PROJECT_ROOT / result["predictions"]) == result["predictions_sha256"],
    }
    access = result["data_access"]
    integrity = {
        "one_paired_development_run": access["paired_development_runs"] == 1,
        "no_evaluation_selection": access["selection_on_development_evaluation"] == 0,
        "no_v44_records_read_during_evaluation": access["v44_records_read_during_evaluation"] == 0,
        "no_model_access": access["model_forward_passes"] == 0,
        "no_training": access["adapter_training_runs"] == 0,
        "implementation_bound": seal["implementation_lock_sha256"] == file_sha256(implementation_path),
        "seal_bound": result["corpus_seal_sha256"] == file_sha256(seal_path),
        "non_final": seal["authorization"]["final_evaluations"] == 0,
    }
    errors = []
    if not all(reproduction.values()):
        errors.append("V45 result does not reproduce")
    if not all(integrity.values()):
        errors.append("V45 firewall or lock chain failed")
    audit = {
        "schema_version": 45,
        "experiment": "v45_post_result_audit",
        "passed": not errors,
        "decision": "accept_v45_paired_development" if not errors else "reject_v45_paired_development",
        "errors": errors,
        "result": str(result_path.relative_to(PROJECT_ROOT)),
        "result_sha256": file_sha256(result_path),
        "reproduction_checks": reproduction,
        "integrity_checks": integrity,
        "scientific_decision": result["decision"],
    }
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    metrics = result["metrics"]
    lines = [
        "# V45 results: declared delayed-language grounding", "",
        f"Decision: `{result['decision']}`.", "",
        "V45 is a paired, non-final development result over the sealed V44 delayed-effect cases.", "",
        "| Metric | Result |", "|---|---:|",
        f"| State-clause exact parse ({metrics['state_clauses']} clauses) | {metrics['state_clause_exact_parse']:.3f} |",
        f"| Canonical state-graph exact ({metrics['state_graphs']} graphs) | {metrics['canonical_state_graph_exact']:.3f} |",
        f"| Bound-action exact parse ({metrics['bound_action_commands']} commands) | {metrics['action_command_exact_parse']:.3f} |",
        f"| Wait-command exact parse ({metrics['wait_commands']} commands) | {metrics['wait_command_exact_parse']:.3f} |",
        f"| Action-sequence exact | {metrics['action_sequence_exact']:.3f} |",
        f"| Safety abstention ({metrics['safety_challenges']} challenges) | {metrics['safety_abstention']:.3f} |",
        f"| Compiled schema recovery | {metrics['compiled_schema_recovery']:.3f} |",
        f"| Compiled next-state exact | {metrics['compiled_next_state_exact']:.3f} |",
        f"| Compiled final-observation exact | {metrics['compiled_final_observation_exact']:.3f} |",
        f"| Compiled wait counterfactual accuracy | {metrics['compiled_wait_counterfactual_accuracy']:.3f} |",
        f"| Collapsed-delay control | {metrics['collapsed_delay_final_exact']:.3f} |",
        f"| End-flush control | {metrics['end_flush_final_exact']:.3f} |",
        f"| Literal language lookup | {metrics['literal_language_lookup_final_exact']:.3f} |", "",
        f"All preregistered gates passed: `{str(result['qualification']['passed']).lower()}`.", "",
        "A pass establishes exact composition of the declared state/action/wait interface with the frozen V44 queue reasoner on paired cases. It is not an independent mechanic-population replication or an open-language result.", "",
        f"Post-result integrity audit: `{'pass' if audit['passed'] else 'fail'}`.",
    ]
    markdown_path.write_text("\n".join(lines) + "\n")
    print(json.dumps(audit, indent=2, sort_keys=True))
    if not audit["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
