#!/usr/bin/env python3
from __future__ import annotations

import json

from audit_and_freeze_v165_factored_ontology_identifiability_population import valid_lock
from v10_protocol import file_sha256
from v193_shadow_menu_interface_frontier import audit_interface, build_interface
from v22r2_grounding import PROJECT_ROOT


def write_json(path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def main() -> None:
    lock_path = PROJECT_ROOT / "configs/v193-shadow-menu-interface-frontier-lock.json"
    lock = json.loads(lock_path.read_text())
    if not valid_lock(lock):
        raise RuntimeError("invalid V193 design lock")
    for key in (
        "config",
        "parent_V192_outcome",
        "source_V190_outcome",
        "source_V186_outcome",
        "source_V186_codebook_lock",
        "contract_catalog",
        "development_bindings",
        "protocol",
        "runner",
    ):
        if file_sha256(PROJECT_ROOT / lock[key]) != lock[f"{key}_sha256"]:
            raise RuntimeError(f"V193 locked dependency changed: {key}")
    output_root = PROJECT_ROOT / "outputs/v193-shadow-menu-interface-frontier/interface"
    if output_root.exists():
        raise RuntimeError("V193 interface output already exists")
    interface = build_interface(
        json.loads((PROJECT_ROOT / lock["contract_catalog"]).read_text()),
        json.loads((PROJECT_ROOT / lock["development_bindings"]).read_text()),
        lock["config_payload"],
    )
    audit = audit_interface(interface, lock["config_payload"])
    config = lock["config_payload"]
    decision = (
        config["decisionRule"]["ifEveryInterfaceParserSafetyAndOracleFrontierGatePasses"]
        if audit["passed"]
        else config["decisionRule"]["otherwise"]
    )
    result = {
        "schema_version": "193-shadow-menu-interface-frontier-result",
        "experiment": lock["experiment"],
        "passed": audit["passed"],
        "decision": decision,
        "claim_boundary": config["claimBoundary"],
        "checks": audit["checks"],
        "summary": audit["summary"],
        "authorization": {
            "preregister_one_deterministic_language_ranker_evaluation_only": bool(audit["passed"]),
            "immediate_language_scoring_or_model_run": False,
            "protected_access_registration_authority_action_or_execution": False,
        },
    }
    write_json(output_root / "visible-menu.json", interface["visible_menu"])
    write_json(output_root / "hidden-option-map.json", interface["hidden_option_map"])
    write_json(output_root / "primary-prior.json", interface["prior"])
    write_json(output_root / "fixed-hierarchy-target-costs.json", interface["fixed_costs"])
    write_json(output_root / "recall-cost-frontier.json", interface["frontier"])
    write_json(output_root / "interface-summary.json", interface["summary"])
    write_json(output_root / "result.json", result)
    print(json.dumps(result, indent=2, sort_keys=True))
    if not audit["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
