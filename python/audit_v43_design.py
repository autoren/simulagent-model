#!/usr/bin/env python3
"""Audit the paired V43 sequential-language design before implementation."""

from __future__ import annotations

import argparse
import json

from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/v43-sequential-language-grounding.json")
    parser.add_argument("--output", default="outputs/v43-sequential-language-grounding/design-audit.json")
    args = parser.parse_args()
    config_path = (PROJECT_ROOT / args.config).resolve()
    output = (PROJECT_ROOT / args.output).resolve()
    config = json.loads(config_path.read_text())
    source_path = PROJECT_ROOT / config["sourceV42OutcomeLock"]
    source = json.loads(source_path.read_text()) if source_path.is_file() else {}
    seal_path = PROJECT_ROOT / config["sourceV42CorpusSeal"]
    seal = json.loads(seal_path.read_text()) if seal_path.is_file() else {}
    implementation_path = PROJECT_ROOT / seal.get("implementation_lock", "missing")
    implementation = json.loads(implementation_path.read_text()) if implementation_path.is_file() else {}
    errors = []
    if not source.get("qualification_passed") or not source.get("authorization", {}).get("preregister_sequential_language_grounding"):
        errors.append("V42 does not authorize V43 preregistration")
    paired = config["pairedDesign"]
    expected = source.get("metrics", {})
    if (paired["mechanics"], paired["querySequences"]) != (expected.get("mechanics"), expected.get("queries")):
        errors.append("V43 paired counts do not match frozen V42")
    if paired["supportSequences"] != implementation.get("expected_counts", {}).get("support_sequences"):
        errors.append("V43 support count is not the frozen V42 count")
    if not paired["reuseSealedV42MechanicsAndCases"] or not paired["noV42TargetDrivenSelection"]:
        errors.append("V43 does not isolate the representation boundary")
    if config["frozenReasoning"]["reasonerModification"] != "forbidden":
        errors.append("V42 reasoner is not frozen")
    if config["languageInterface"]["openParaphrase"]:
        errors.append("V43 expands beyond declared controlled language")
    if config["nextAxisIfPassed"]["axis"] != "deterministic_delayed_effects" or not config["nextAxisIfPassed"]["stochasticityStillDeferred"]:
        errors.append("V43 does not isolate the next temporal axis")
    gates = config["gates"]
    if gates["maximumBagOfActionsOrderCounterfactualAccuracy"] >= 1.0 or gates["maximumLiteralLanguageLookupFinalExact"] >= 1.0:
        errors.append("V43 inadequacy controls are not required to fail")
    downstream = (
        "configs/v43-design-lock.json",
        "configs/v43-implementation-lock.json",
        "configs/v43-corpus-seal.json",
        "data/v43-sequential-language-grounding",
        "outputs/v43-sequential-language-grounding/development",
    )
    if any((PROJECT_ROOT / path).exists() for path in downstream):
        errors.append("V43 downstream artifact exists before design lock")
    audit = {
        "schema_version": 43,
        "experiment": "v43_design_audit",
        "passed": not errors,
        "decision": "authorize_v43_design_lock" if not errors else "repair_v43_design",
        "errors": errors,
        "config": str(config_path.relative_to(PROJECT_ROOT)),
        "config_sha256": file_sha256(config_path),
        "source_v42_outcome_lock": str(source_path.relative_to(PROJECT_ROOT)),
        "source_v42_outcome_lock_sha256": file_sha256(source_path) if source_path.is_file() else None,
        "source_v42_corpus_seal": str(seal_path.relative_to(PROJECT_ROOT)),
        "source_v42_corpus_seal_sha256": file_sha256(seal_path) if seal_path.is_file() else None,
        "checks": {
            "paired_representation_isolation": paired["reuseSealedV42MechanicsAndCases"],
            "v42_reasoner_frozen": config["frozenReasoning"]["reasonerModification"] == "forbidden",
            "state_and_action_language_registered": set(config["languageInterface"]["actionOrientations"]) == {"actor_first", "target_first"},
            "safety_suite_registered": all(config["safetyChallenges"].values()),
            "non_final": config["firewall"]["finalEvaluation"] == "forbidden",
            "no_model_access": config["firewall"]["languageModelAccess"] == "forbidden",
        },
        "data_access": {
            "v42_records_read": 0,
            "paired_development_runs": 0,
            "language_model_forward_passes": 0,
            "adapter_training_runs": 0,
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    print(json.dumps(audit, indent=2, sort_keys=True))
    if not audit["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
