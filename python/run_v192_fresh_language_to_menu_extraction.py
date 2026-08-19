#!/usr/bin/env python3
from __future__ import annotations

import json

from audit_and_freeze_v165_factored_ontology_identifiability_population import valid_lock
from v10_protocol import file_sha256
from v192_fresh_language_to_menu_extraction import audit_extraction, build_extraction
from v22r2_grounding import PROJECT_ROOT


def write_json(path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def main() -> None:
    lock_path = PROJECT_ROOT / "configs/v192-fresh-language-to-menu-extraction-lock.json"
    lock = json.loads(lock_path.read_text())
    if not valid_lock(lock):
        raise RuntimeError("invalid V192 design lock")
    for key in (
        "config",
        "parent_V191_outcome",
        "parent_V191_population_lock",
        "source_archive",
        "development_identities",
        "hidden_targets",
        "protocol",
        "runner",
    ):
        if file_sha256(PROJECT_ROOT / lock[key]) != lock[f"{key}_sha256"]:
            raise RuntimeError(f"V192 locked dependency changed: {key}")
    output_root = PROJECT_ROOT / "outputs/v192-fresh-language-to-menu-extraction/extraction"
    if output_root.exists():
        raise RuntimeError("V192 extraction output already exists")
    extraction = build_extraction(
        (PROJECT_ROOT / lock["source_archive"]).read_bytes(),
        json.loads((PROJECT_ROOT / lock["development_identities"]).read_text()),
        json.loads((PROJECT_ROOT / lock["hidden_targets"]).read_text()),
        lock["config_payload"],
    )
    audit = audit_extraction(extraction, lock["config_payload"])
    decision = (
        lock["config_payload"]["decisionRule"]["ifEveryExtractionProjectionIsolationAndAccessGatePasses"]
        if audit["passed"]
        else lock["config_payload"]["decisionRule"]["otherwise"]
    )
    result = {
        "schema_version": "192-fresh-language-to-menu-extraction-result",
        "experiment": lock["experiment"],
        "passed": audit["passed"],
        "decision": decision,
        "claim_boundary": lock["config_payload"]["claimBoundary"],
        "checks": audit["checks"],
        "summary": audit["summary"],
        "authorization": {
            "preregister_shadow_menu_interface_and_oracle_frontier_only": bool(audit["passed"]),
            "immediate_interface_scoring_or_model_run": False,
            "protected_access_registration_authority_action_or_execution": False,
        },
    }
    write_json(output_root / "development-language.json", extraction["development_language"])
    write_json(output_root / "extraction-summary.json", extraction["summary"])
    write_json(output_root / "result.json", result)
    print(json.dumps(result, indent=2, sort_keys=True))
    if not audit["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
