#!/usr/bin/env python3
"""Reproduce, audit, and summarize the V42 oracle development result."""

from __future__ import annotations

import argparse
import json

from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT
from evaluate_v42_sequential import aggregate, evaluate_record, qualification, read
from v42_stateful import mechanic_registry


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", default="outputs/v42-sequential-state-foundation/development/result.json")
    parser.add_argument("--audit", default="outputs/v42-sequential-state-foundation/post-result-audit.json")
    parser.add_argument("--markdown", default="docs/v42-results.md")
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
        "one_oracle_development_run": access["oracle_development_runs"] == 1,
        "no_evaluation_selection": access["selection_on_development_evaluation"] == 0,
        "no_language_model_access": access["language_model_forward_passes"] == 0,
        "no_adapter_training": access["adapter_training_runs"] == 0,
        "no_v41_record_reuse": access["v41_records_read"] == 0,
        "implementation_bound": seal["implementation_lock_sha256"] == file_sha256(implementation_path),
        "seal_bound": result["corpus_seal_sha256"] == file_sha256(seal_path),
        "non_final": seal["authorization"]["final_evaluations"] == 0,
    }
    errors = []
    if not all(reproduction.values()):
        errors.append("V42 result does not reproduce")
    if not all(integrity.values()):
        errors.append("V42 firewall or lock chain failed")
    audit = {
        "schema_version": 42,
        "experiment": "v42_post_result_audit",
        "passed": not errors,
        "decision": "accept_v42_oracle_development" if not errors else "reject_v42_oracle_development",
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
        "# V42 results: oracle sequential-state foundation",
        "",
        f"Decision: `{result['decision']}`.",
        "",
        "V42 is an oracle-first, language-free development result. It isolates persistent deterministic state mutation across ordered action sequences.",
        "",
        "| Metric | Result |",
        "|---|---:|",
        f"| Oracle program validation | {metrics['oracle_program_validation']:.3f} |",
        f"| Stateful target retention | {metrics['stateful_target_retention']:.3f} |",
        f"| Stateful schema recovery | {metrics['stateful_schema_recovery']:.3f} |",
        f"| Stateful next-state exact | {metrics['stateful_next_state_exact']:.3f} |",
        f"| Stateful final-observation exact | {metrics['stateful_final_observation_exact']:.3f} |",
        f"| Stateful complete-mechanic exact | {metrics['stateful_complete_mechanic_exact']:.3f} |",
        f"| Order-counterfactual accuracy | {metrics['order_counterfactual_accuracy']:.3f} |",
        f"| Memoryless oracle-program control | {metrics['memoryless_final_observation_exact']:.3f} |",
        f"| Literal sequence lookup | {metrics['literal_lookup_final_observation_exact']:.3f} |",
        "",
        f"All preregistered development gates passed: `{str(result['qualification']['passed']).lower()}`.",
        "",
        "Interpretation: if accepted, persistent state is both representable and necessary in this controlled benchmark. The result authorizes a separately preregistered sequential-language grounding experiment; it does not yet authorize stochasticity, delays, active intervention choice, open ontologies, or a final evaluation.",
        "",
        f"Post-result integrity audit: `{'pass' if audit['passed'] else 'fail'}`.",
    ]
    markdown_path.write_text("\n".join(lines) + "\n")
    print(json.dumps(audit, indent=2, sort_keys=True))
    if not audit["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
