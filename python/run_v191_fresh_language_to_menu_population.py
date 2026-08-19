#!/usr/bin/env python3
from __future__ import annotations

import json

from audit_and_freeze_v165_factored_ontology_identifiability_population import valid_lock
from v10_protocol import file_sha256
from v191_fresh_language_to_menu_population import audit_population, build_population
from v22r2_grounding import PROJECT_ROOT


def write_json(path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def main() -> None:
    lock_path = PROJECT_ROOT / "configs/v191-fresh-language-to-menu-population-lock.json"
    lock = json.loads(lock_path.read_text())
    if not valid_lock(lock):
        raise RuntimeError("invalid V191 design lock")
    for key in (
        "config",
        "parent_V190_outcome",
        "source_V124_outcome",
        "source_V183_outcome",
        "source_inventory",
        "contract_catalog",
        "previous_hidden_population",
        "protocol",
        "runner",
    ):
        if file_sha256(PROJECT_ROOT / lock[key]) != lock[f"{key}_sha256"]:
            raise RuntimeError(f"V191 locked dependency changed: {key}")
    output_root = PROJECT_ROOT / "outputs/v191-fresh-language-to-menu-population/population"
    if output_root.exists():
        raise RuntimeError("V191 population output already exists")

    config = lock["config_payload"]
    population = build_population(
        json.loads((PROJECT_ROOT / lock["source_inventory"]).read_text()),
        json.loads((PROJECT_ROOT / lock["contract_catalog"]).read_text()),
        json.loads((PROJECT_ROOT / lock["previous_hidden_population"]).read_text()),
        config,
    )
    audit = audit_population(population, config)
    decision = (
        config["decisionRule"]["ifEveryFreshnessAvailabilityPopulationAndAccessGatePasses"]
        if audit["passed"]
        else config["decisionRule"]["otherwise"]
    )
    result = {
        "schema_version": "191-fresh-language-to-menu-population-result",
        "experiment": config["experiment"],
        "passed": audit["passed"],
        "decision": decision,
        "claim_boundary": config["claimBoundary"],
        "checks": audit["checks"],
        "summary": audit["summary"],
        "authorization": {
            "preregister_separate_exact_development_language_extraction_only": bool(audit["passed"]),
            "immediate_language_extraction_interface_scoring_or_model_run": False,
            "protected_access_registration_authority_action_or_execution": False,
        },
    }
    write_json(output_root / "availability-census.json", population["availability_census"])
    write_json(output_root / "fresh-development-identities.json", population["public_identities"])
    write_json(output_root / "hidden-targets.json", population["hidden_targets"])
    write_json(output_root / "population-summary.json", population["summary"])
    write_json(output_root / "result.json", result)
    print(json.dumps(result, indent=2, sort_keys=True))
    if not audit["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
