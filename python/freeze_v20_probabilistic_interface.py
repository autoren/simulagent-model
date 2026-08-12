"""Freeze the V20 development evaluation after the V21 design is sealed."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from v10_protocol import file_sha256


PROJECT_ROOT = Path(__file__).resolve().parents[1]
IMPLEMENTATION = (
    "python/v20_probabilistic_grounding.py",
    "python/test_v20_probabilistic.py",
    "python/audit_v20_protocol.py",
    "python/evaluate_v20_probabilistic_interface.py",
    "python/freeze_v20_probabilistic_interface.py",
    "python/evaluate_v19_frozen_integration.py",
    "python/evaluate_v15_full_pipeline.py",
    "python/evaluate_v10_frozen.py",
    "python/run_v18_schema_baselines.py",
    "python/v18_schema.py",
    "python/v10_protocol.py",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/v20-probabilistic-interface.json")
    parser.add_argument("--plan", default="docs/v20-probabilistic-interface-plan.md")
    parser.add_argument("--audit", default="outputs/v20-probabilistic-interface/pre-evaluation-audit.json")
    parser.add_argument("--output", default="configs/v20-probabilistic-interface-lock.json")
    args = parser.parse_args()
    output = PROJECT_ROOT / args.output
    if output.exists():
        raise RuntimeError(f"V20 protocol lock already exists: {output}")
    if (PROJECT_ROOT / "outputs/v20-probabilistic-interface/evaluation").exists():
        raise RuntimeError("V20 evaluation exists before its protocol lock")
    if (PROJECT_ROOT / "data/v21-final").exists():
        raise RuntimeError("V21 final records exist before V20 is frozen")
    design_lock_path = PROJECT_ROOT / "configs/v21-multimechanic-design-lock.json"
    if not design_lock_path.exists():
        raise RuntimeError("V21 unmaterialized design must be locked before V20 evaluation")
    audit_path = PROJECT_ROOT / args.audit
    audit = json.loads(audit_path.read_text())
    if not audit["passed"]:
        raise RuntimeError("V20 pre-evaluation audit did not pass")
    config_path = PROJECT_ROOT / args.config
    config = json.loads(config_path.read_text())
    v19_lock_path = PROJECT_ROOT / config["sourceV19Lock"]
    v19_lock = json.loads(v19_lock_path.read_text())
    metadata_path = PROJECT_ROOT / config["sourceV19Features"]
    metadata = json.loads(metadata_path.read_text())
    feature_path = PROJECT_ROOT / metadata["feature_artifact"]
    source_paths = {
        "v19_lock": v19_lock_path,
        "v19_feature_metadata": metadata_path,
        "v19_feature_artifact": feature_path,
        "v19_result": PROJECT_ROOT / config["sourceV19Result"],
        "v19_post_audit": PROJECT_ROOT / config["sourceV19PostAudit"],
        "v19_correction": PROJECT_ROOT / config["sourceV19Correction"],
        "deployment_heads": PROJECT_ROOT / config["sourceDeploymentHeads"],
        "v21_design_lock": design_lock_path,
    }
    lock = {
        "schema_version": 20,
        "experiment": config["experiment"],
        "config": config,
        "config_path": args.config,
        "config_sha256": file_sha256(config_path),
        "plan": args.plan,
        "plan_sha256": file_sha256(PROJECT_ROOT / args.plan),
        "pre_evaluation_audit": args.audit,
        "pre_evaluation_audit_sha256": file_sha256(audit_path),
        "implementation": {
            path: file_sha256(PROJECT_ROOT / path) for path in IMPLEMENTATION
        },
        "source": {
            f"{name}_sha256": file_sha256(path) for name, path in source_paths.items()
        } | {
            "v19_dataset": v19_lock["source"]["v19_dataset"],
            "v18_dataset": v19_lock["source"]["v18_dataset"],
            "v19_feature_artifact": metadata["feature_artifact"],
        },
        "limits": config["limits"],
        "evaluation_count_before_lock": 0,
        "final_records_before_lock": 0,
    }
    output.write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "output": args.output,
        "lock_sha256": file_sha256(output),
        "implementation_files": len(lock["implementation"]),
        "v21_design_lock_sha256": lock["source"]["v21_design_lock_sha256"],
        "evaluation_count_before_lock": 0,
        "final_records_before_lock": 0,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
