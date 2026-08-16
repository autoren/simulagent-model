#!/usr/bin/env python3
"""Seal the audited V36 corpus and authorize its one feature extraction."""

from __future__ import annotations

import argparse
import hashlib
import json

from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--interface-lock", default="configs/v36-interface-lock.json")
    parser.add_argument("--manifest", default="data/v36-independent-confirmation/manifest.json")
    parser.add_argument("--audit", default="outputs/v36-independent-confirmation/corpus-audit.json")
    parser.add_argument("--output", default="configs/v36-confirmation-seal.json")
    args = parser.parse_args()
    interface_path, manifest_path, audit_path, output_path = map(lambda value: (PROJECT_ROOT / value).resolve(), (args.interface_lock, args.manifest, args.audit, args.output))
    if output_path.exists():
        raise RuntimeError("V36 confirmation seal already exists")
    interface, manifest, audit = (json.loads(path.read_text()) for path in (interface_path, manifest_path, audit_path))
    if not audit["passed"] or audit["decision"] != "authorize_v36_confirmation_seal" or audit["source"]["manifest_sha256"] != file_sha256(manifest_path):
        raise RuntimeError("V36 corpus audit does not authorize sealing")
    artifact_path = PROJECT_ROOT / manifest["artifact"]
    seal = {
        "schema_version": 36, "experiment": "v36_confirmation_seal",
        "interface_lock": str(interface_path.relative_to(PROJECT_ROOT)), "interface_lock_sha256": file_sha256(interface_path),
        "implementation_lock": interface["implementation_lock"], "implementation_lock_sha256": interface["implementation_lock_sha256"],
        "manifest": str(manifest_path.relative_to(PROJECT_ROOT)), "manifest_sha256": file_sha256(manifest_path),
        "corpus_artifact": str(artifact_path.relative_to(PROJECT_ROOT)), "corpus_artifact_sha256": file_sha256(artifact_path),
        "corpus_audit": str(audit_path.relative_to(PROJECT_ROOT)), "corpus_audit_sha256": file_sha256(audit_path),
        "population": audit["population"], "pair_counts": audit["pair_counts"],
        "authorization": {"feature_extraction": True, "feature_extractions": 1, "backbone_forward_passes": 3510, "evaluate_confirmation": False, "reuse_v32_evaluation": False, "run_v28": False, "construct_final_suite": False},
    }
    seal["lock_payload_sha256"] = hashlib.sha256(json.dumps(seal, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    output_path.write_text(json.dumps(seal, indent=2, sort_keys=True) + "\n")
    print(json.dumps(seal, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
