#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json

from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--population", default="data/v57-definition-augmented-ontology-transfer")
    parser.add_argument("--output", default="outputs/v57-definition-augmented-ontology-transfer/population-audit.json")
    args = parser.parse_args()
    population = (PROJECT_ROOT / args.population).resolve()
    output = (PROJECT_ROOT / args.output).resolve()
    manifest_path = population / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    lock_path = PROJECT_ROOT / manifest["implementation_lock"]
    lock = json.loads(lock_path.read_text())
    config = lock["config_payload"]
    errors: list[str] = []

    binding_ok = (
        lock["authorization"]["audit_and_seal_v57_population"]
        and not lock["authorization"]["run_v57_candidate_evaluation"]
        and manifest["implementation_lock_sha256"] == file_sha256(lock_path)
        and all(
            file_sha256(PROJECT_ROOT / path) == digest
            for path, digest in lock["implementation_files_sha256"].items()
        )
    )
    if not binding_ok:
        errors.append("V57 population is not bound to its locked implementation")

    artifacts_ok = True
    populations = {}
    for name, artifact in manifest["artifacts"].items():
        path = PROJECT_ROOT / artifact["path"]
        rows = read_jsonl(path)
        populations[name] = rows
        artifacts_ok = artifacts_ok and (
            len(rows) == artifact["records"]
            and file_sha256(path) == artifact["sha256"]
        )
    if not artifacts_ok:
        errors.append("V57 population artifact hashes or counts changed")

    core, safety = populations["core"], populations["safety"]
    core_schema_ok = (
        len(core) == config["population"]["coreRecords"]
        and len({row["id"] for row in core}) == len(core)
        and all(
            set(row) == {
                "id", "schema_version", "split", "ontology_pack", "public",
                "target", "oracle_metadata",
            }
            and row["schema_version"] == 57 and row["split"] == "core"
            and set(row["public"]) == {"entities", "concept_definitions", "evidence_text"}
            and "target" not in row["public"] and "oracle_metadata" not in row["public"]
            for row in core
        )
    )
    if not core_schema_ok:
        errors.append("V57 core IDs, schema, or public/target firewall is invalid")

    packs = {
        pack: [row for row in core if row["ontology_pack"] == pack]
        for pack in {row["ontology_pack"] for row in core}
    }
    kinds = {row["oracle_metadata"]["concept_kind"] for row in core}
    families = {row["oracle_metadata"]["definition_template_family"] for row in core}
    factor_ok = (
        len(packs) == config["population"]["ontologyPacks"]
        and set(map(len, packs.values())) == {config["population"]["coreRecordsPerPack"]}
        and kinds == set(config["population"]["factors"]["conceptKind"])
        and families == set(config["population"]["factors"]["definitionTemplateFamily"])
        and all(len(row["public"]["concept_definitions"]) == 6 for row in core)
        and all(
            concept["opaque_id"].startswith("sym_")
            for row in core for concept in row["public"]["concept_definitions"]
        )
    )
    if not factor_ok:
        errors.append("V57 pack, kind, template, or opaque-symbol coverage is invalid")

    conditions = {row["expected"]["condition"] for row in safety}
    safety_ok = (
        len(safety) == config["safetyPopulation"]["records"]
        and len({row["id"] for row in safety}) == len(safety)
        and conditions == set(config["safetyPopulation"]["conditions"])
        and all(
            set(row) == {
                "id", "schema_version", "split", "ontology_pack", "public",
                "expected", "oracle_metadata",
            }
            and row["split"] == "safety" and "target" not in row["public"]
            for row in safety
        )
        and all(
            sum(row["expected"]["condition"] == condition for row in safety)
            == config["population"]["ontologyPacks"]
            * config["safetyPopulation"]["recordsPerConditionPerPack"]
            for condition in conditions
        )
    )
    if not safety_ok:
        errors.append("V57 safety census, conditions, or public firewall is invalid")

    integrity_ok = (
        manifest["truth_access_count"] == 0
        and manifest["evaluation_runs"] == 0
        and not any(
            (PROJECT_ROOT / path).exists()
            for path in (
                "configs/v57-population-seal.json", "configs/v57-evaluation-implementation-lock.json",
                "configs/v57-outcome-lock.json",
                "outputs/v57-definition-augmented-ontology-transfer/evaluation-attempt.json",
                "outputs/v57-definition-augmented-ontology-transfer/evaluation",
            )
        )
    )
    if not integrity_ok:
        errors.append("V57 truth, evaluation, or downstream firewall failed")

    audit = {
        "schema_version": 57,
        "experiment": "v57_population_audit",
        "passed": not errors,
        "decision": "authorize_v57_population_seal" if not errors else "repair_v57_population",
        "errors": errors,
        "population": str(population.relative_to(PROJECT_ROOT)),
        "manifest": str(manifest_path.relative_to(PROJECT_ROOT)),
        "manifest_sha256": file_sha256(manifest_path),
        "implementation_lock": str(lock_path.relative_to(PROJECT_ROOT)),
        "implementation_lock_sha256": file_sha256(lock_path),
        "checks": {
            "implementation_authorization_and_binding": binding_ok,
            "artifact_hashes_and_counts": artifacts_ok,
            "core_ids_schema_and_public_target_firewall": core_schema_ok,
            "pack_kind_template_and_opaque_symbol_coverage": factor_ok,
            "safety_census_conditions_and_firewall": safety_ok,
            "truth_evaluation_and_downstream_firewall": integrity_ok,
        },
        "metrics": {
            "core_records": len(core), "safety_records": len(safety),
            "ontology_packs": len(packs), "concept_kinds": sorted(kinds),
            "definition_template_families": sorted(families),
            "truth_access_count": manifest["truth_access_count"],
            "evaluation_runs": manifest["evaluation_runs"],
        },
    }
    output.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    print(json.dumps(audit, indent=2, sort_keys=True))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
