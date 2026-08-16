#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json

from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/v59-budgeted-root-sampled-planning.json")
    parser.add_argument("--plan", default="docs/v59-budgeted-root-sampled-planning-plan.md")
    parser.add_argument("--deferral", default="docs/v58-deferred-status.md")
    parser.add_argument("--audit", default="outputs/v59-budgeted-root-sampled-planning/design-audit.json")
    parser.add_argument("--output", default="configs/v59-design-lock.json")
    args = parser.parse_args()
    config_path, plan_path, deferral_path, audit_path, output = tuple(
        (PROJECT_ROOT / value).resolve()
        for value in (args.config, args.plan, args.deferral, args.audit, args.output)
    )
    if output.exists():
        raise RuntimeError("V59 design already frozen")
    audit = json.loads(audit_path.read_text())
    source_hashes = {
        path: file_sha256(PROJECT_ROOT / path)
        for path in audit["source_outcome_locks_sha256"]
    }
    if (
        not audit["passed"]
        or audit["config_sha256"] != file_sha256(config_path)
        or audit["preregistration_sha256"] != file_sha256(plan_path)
        or audit["v58_deferral_sha256"] != file_sha256(deferral_path)
        or audit["source_outcome_locks_sha256"] != source_hashes
    ):
        raise RuntimeError("V59 design audit is not intact and bound")
    lock = {
        "schema_version": 59,
        "experiment": "v59_design_lock",
        "config": str(config_path.relative_to(PROJECT_ROOT)),
        "config_sha256": file_sha256(config_path),
        "preregistration": str(plan_path.relative_to(PROJECT_ROOT)),
        "preregistration_sha256": file_sha256(plan_path),
        "v58_deferral": str(deferral_path.relative_to(PROJECT_ROOT)),
        "v58_deferral_sha256": file_sha256(deferral_path),
        "design_audit": str(audit_path.relative_to(PROJECT_ROOT)),
        "design_audit_sha256": file_sha256(audit_path),
        "source_outcome_locks_sha256": source_hashes,
        "config_payload": json.loads(config_path.read_text()),
        "authorization": {
            "write_and_audit_candidate_search": True,
            "write_and_audit_independent_generator": True,
            "construct_v59_population": False,
            "run_v59_evaluation": False,
            "simulate_human_v58_records": False,
            "model_access": False,
        },
    }
    lock["lock_payload_sha256"] = hashlib.sha256(
        json.dumps(lock, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    output.write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n")
    print(json.dumps(lock, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

