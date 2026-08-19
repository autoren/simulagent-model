#!/usr/bin/env python3
from __future__ import annotations

import json

from audit_and_freeze_v165_factored_ontology_identifiability_population import valid_lock
from v10_protocol import file_sha256
from v199_exact_menu_presentation_robustness import audit_transformation_family, build_transformation_family
from v22r2_grounding import PROJECT_ROOT


def write_json(path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def main() -> None:
    lock_path = PROJECT_ROOT / "configs/v199-exact-menu-presentation-robustness-lock.json"
    lock = json.loads(lock_path.read_text())
    if not valid_lock(lock):
        raise RuntimeError("invalid V199 design lock")
    for key in (
        "config", "parent_V198_outcome", "source_V195_outcome", "source_V193_outcome", "source_V191_outcome",
        "development_identities", "hidden_targets", "canonical_visible_menu", "canonical_hidden_option_map",
        "roadmap", "plan", "protocol", "tests", "runner", "verifier", "auditor", "design_audit",
    ):
        if file_sha256(PROJECT_ROOT / lock[key]) != lock[f"{key}_sha256"]:
            raise RuntimeError(f"V199 locked dependency changed: {key}")
    output_root = PROJECT_ROOT / "outputs/v199-exact-menu-presentation-robustness/census"
    if output_root.exists():
        raise RuntimeError("V199 census output already exists")
    config = lock["config_payload"]
    family = build_transformation_family(
        json.loads((PROJECT_ROOT / lock["development_identities"]).read_text()),
        json.loads((PROJECT_ROOT / lock["hidden_targets"]).read_text()),
        json.loads((PROJECT_ROOT / lock["canonical_visible_menu"]).read_text()),
        json.loads((PROJECT_ROOT / lock["canonical_hidden_option_map"]).read_text()),
        config,
    )
    audit = audit_transformation_family(family, config)
    decision = (
        config["decisionRule"]["ifEveryExactFeasibilityAndAccessGatePasses"]
        if audit["passed"] else config["decisionRule"]["otherwise"]
    )
    result = {
        "schema_version": "199-exact-menu-presentation-robustness-result",
        "experiment": config["experiment"],
        "passed": audit["passed"],
        "decision": decision,
        "claim_boundary": config["claimBoundary"],
        "checks": audit["checks"],
        "summary": audit["summary"],
        "future_paired_development_gates": config["futurePairedDevelopmentGates"],
        "authorization": {
            "preregister_separate_deterministic_development_evaluation_only": bool(audit["passed"]),
            "immediate_language_scoring_or_model_run": False,
            "protected_API_training_registration_authority_action_or_execution": False,
        },
    }
    write_json(output_root / "visible-menu-variants.json", family["visible_variants"])
    write_json(output_root / "hidden-variant-maps.json", family["hidden_variant_maps"])
    write_json(output_root / "transformation-audit-rows.json", family["audit_rows"])
    write_json(output_root / "summary.json", family["summary"])
    write_json(output_root / "result.json", result)
    print(json.dumps(result, indent=2, sort_keys=True))
    if not audit["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
