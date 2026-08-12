"""Freeze the V21 generator and preregistration before V20 evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from v10_protocol import file_sha256


PROJECT_ROOT = Path(__file__).resolve().parents[1]
IMPLEMENTATION = (
    "python/v21_final_suite.py",
    "python/test_v21_final_suite.py",
    "python/audit_v21_design.py",
    "python/freeze_v21_design.py",
    "python/materialize_v21_final_suite.py",
    "python/v18_schema.py",
    "python/generate_v18_schema_benchmark.py",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/v21-multimechanic-final.json")
    parser.add_argument("--plan", default="docs/v21-multimechanic-final-plan.md")
    parser.add_argument("--charter", default="docs/research-direction.md")
    parser.add_argument("--audit", default="outputs/v21-design/pre-materialization-audit.json")
    parser.add_argument("--output", default="configs/v21-multimechanic-design-lock.json")
    args = parser.parse_args()
    output = PROJECT_ROOT / args.output
    if output.exists():
        raise RuntimeError(f"V21 design lock already exists: {output}")
    if (PROJECT_ROOT / "outputs/v20-probabilistic-interface/evaluation").exists():
        raise RuntimeError("V21 design must be locked before V20 evaluation")
    if (PROJECT_ROOT / "data/v21-final").exists():
        raise RuntimeError("V21 final data exists before the design lock")
    audit_path = PROJECT_ROOT / args.audit
    audit = json.loads(audit_path.read_text())
    if not audit["passed"]:
        raise RuntimeError("V21 design audit did not pass")
    config_path = PROJECT_ROOT / args.config
    config = json.loads(config_path.read_text())
    v19_lock_path = PROJECT_ROOT / "configs/v19-frozen-integration-lock.json"
    v19_lock = json.loads(v19_lock_path.read_text())
    lock = {
        "schema_version": 21,
        "experiment": "v21_unmaterialized_population_design_lock",
        "config": config,
        "config_path": args.config,
        "config_sha256": file_sha256(config_path),
        "plan": args.plan,
        "plan_sha256": file_sha256(PROJECT_ROOT / args.plan),
        "research_charter": args.charter,
        "research_charter_sha256": file_sha256(PROJECT_ROOT / args.charter),
        "pre_materialization_audit": args.audit,
        "pre_materialization_audit_sha256": file_sha256(audit_path),
        "implementation": {
            path: file_sha256(PROJECT_ROOT / path) for path in IMPLEMENTATION
        },
        "source": {
            "v18_dataset": v19_lock["source"]["v18_dataset"],
            "v18_dataset_sha256": v19_lock["source"]["v18_dataset_sha256"],
            "v18_manifest": v19_lock["source"]["v18_manifest"],
            "v18_manifest_sha256": v19_lock["source"]["v18_manifest_sha256"],
            "v19_lock": "configs/v19-frozen-integration-lock.json",
            "v19_lock_sha256": file_sha256(v19_lock_path),
            "v20_config": "configs/v20-probabilistic-interface.json",
            "v20_config_sha256": file_sha256(PROJECT_ROOT / "configs/v20-probabilistic-interface.json"),
            "v20_plan": "docs/v20-probabilistic-interface-plan.md",
            "v20_plan_sha256": file_sha256(PROJECT_ROOT / "docs/v20-probabilistic-interface-plan.md"),
        },
        "seed_policy": config["seedPolicy"],
        "limits": config["limits"],
        "data_access_before_lock": {
            "v20_results_read": 0,
            "v17_records_read": 0,
            "v17_model_results_read": 0,
            "final_records_created_or_read": 0,
            "model_forward_passes": 0,
        },
    }
    output.write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "output": args.output,
        "lock_sha256": file_sha256(output),
        "implementation_files": len(lock["implementation"]),
        "final_records": 0,
        "v20_results_read": 0,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
