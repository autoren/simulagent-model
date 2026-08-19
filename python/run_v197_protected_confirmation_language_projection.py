#!/usr/bin/env python3
from __future__ import annotations

import json

from audit_and_freeze_v165_factored_ontology_identifiability_population import valid_lock
from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT
from v197_protected_confirmation_language_projection import audit_projection, build_projection


def write_json(path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def main() -> None:
    lock_path = PROJECT_ROOT / "configs/v197-protected-confirmation-language-projection-lock.json"
    lock = json.loads(lock_path.read_text())
    if not valid_lock(lock):
        raise RuntimeError("invalid V197 lock")
    for key in [key for key in lock if not key.endswith("_sha256") and f"{key}_sha256" in lock]:
        if file_sha256(PROJECT_ROOT / lock[key]) != lock[f"{key}_sha256"]:
            raise RuntimeError(f"V197 dependency drifted: {key}")
    output_root = PROJECT_ROOT / "outputs/v197-protected-confirmation-language-projection/projection"
    if output_root.exists():
        raise RuntimeError("V197 projection already exists")
    config = lock["config_payload"]
    projected = build_projection(
        json.loads((PROJECT_ROOT / lock["sealed_protected_language"]).read_text()),
        json.loads((PROJECT_ROOT / lock["confirmation_identities"]).read_text()),
        config,
    )
    audit = audit_projection(projected, config)
    decision = (
        config["decisionRule"]["ifEveryProjectionSeparationAndAccessGatePasses"]
        if audit["passed"] else config["decisionRule"]["otherwise"]
    )
    result = {
        "schema_version": "197-protected-confirmation-language-projection-result",
        "experiment": config["experiment"], "passed": audit["passed"], "decision": decision,
        "claim_boundary": config["claimBoundary"], "checks": audit["checks"], "summary": audit["summary"],
        "authorization": {
            "preregister_unchanged_V195_policy_confirmation_only": audit["passed"],
            "immediate_model_run": False,
            "API_training_registration_authority_action_or_execution": False,
        },
    }
    write_json(output_root / "confirmation-language.json", projected["language"])
    write_json(output_root / "projection-summary.json", projected["summary"])
    write_json(output_root / "result.json", result)
    print(json.dumps(result, indent=2, sort_keys=True))
    if not audit["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
