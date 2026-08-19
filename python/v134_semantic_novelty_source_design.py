from __future__ import annotations

from collections import Counter
from typing import Any

from v93_open_set_source import canonical_sha256, hash_order
from v133_sgd_capability_label_identifiability import compare_definitions, read_schema_definitions


def derive_classes(inventory: dict[str, Any], partition: str) -> list[dict[str, Any]]:
    train = [row for row in inventory["candidate_index"] if row["partition"] == "train"]
    train_pairs = {(row["service"], row["intent"]) for row in train}
    train_services = {row["service"] for row in train}; train_domains = {row["domain"] for row in train}
    output = []
    for row in inventory["candidate_index"]:
        if row["partition"] != partition: continue
        if (row["service"], row["intent"]) in train_pairs: label = "known"
        elif row["service"] not in train_services and row["domain"] in train_domains: label = "novel_valid"
        elif row["domain"] not in train_domains: label = "unsupported"
        else: label = "other"
        output.append({**row, "derived_class_label": label})
    return output


def build_catalog(rows: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
    choices = []
    for index, pair in enumerate(config["catalog"]["declaredKnownPairs"], start=1):
        service, intent = pair.split("::", 1); domains = {row["domain"] for row in rows if row["service"] == service and row["intent"] == intent}
        if len(domains) != 1: raise ValueError(f"V134 known pair missing or ambiguous: {pair}")
        choices.append({"choice_id": f"K{index:02d}", "kind": "KNOWN", "domain": next(iter(domains)), "service": service, "intent": intent, "intent_id": pair})
    for index, domain in enumerate(config["catalog"]["novelCompositeDomains"], start=1):
        members = sorted({f"{row['service']}::{row['intent']}" for row in rows if row["derived_class_label"] == "novel_valid" and row["domain"] == domain})
        if not members: raise ValueError(f"V134 novel domain missing: {domain}")
        choices.append({"choice_id": f"N{index:02d}", "kind": "NOVEL_COMPOSITE", "domain": domain, "member_intent_ids": members})
    unsupported_domains = sorted({row["domain"] for row in rows if row["derived_class_label"] == "unsupported"})
    choices.append({"choice_id": "U00", "kind": "UNSUPPORTED_COMPOSITE", "domains": unsupported_domains})
    choices.append({"choice_id": "A00", "kind": "ABSTAIN", "meaning": "insufficient evidence"})
    visible_domains = sorted({row.get("domain") for row in choices if row.get("domain")})
    return {"choice_count": len(choices), "choices": choices, "visible_domains": visible_domains, "declared_known_pairs": config["catalog"]["declaredKnownPairs"], "complete_safe_composite_hypothesis_universe": True, "contains_language": False, "catalog_sha256": canonical_sha256(choices)}


def select_population(rows: list[dict[str, Any]], catalog: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    spec = config["population"]; known = [row for row in catalog["choices"] if row["kind"] == "KNOWN"]; novel = [row for row in catalog["choices"] if row["kind"] == "NOVEL_COMPOSITE"]
    candidates = [row["choice_id"] for row in known]; per_truth = spec["recordsPerTruthCandidateCell"] * len(candidates)
    pools: dict[str, list[dict[str, Any]]] = {}
    def take(pool: list[dict[str, Any]], count: int, group: str) -> list[dict[str, Any]]:
        ordered = sorted(pool, key=lambda row: hash_order(spec["baseSalt"], group, row["candidate_id"]))
        if len(ordered) < count: raise ValueError(f"insufficient V134 group {group}: {len(ordered)}")
        return ordered[:count]
    for choice in known:
        pools[choice["choice_id"]] = take([row for row in rows if row["derived_class_label"] == "known" and f"{row['service']}::{row['intent']}" == choice["intent_id"]], per_truth, choice["choice_id"])
    for choice in novel:
        pools[choice["choice_id"]] = take([row for row in rows if row["derived_class_label"] == "novel_valid" and row["domain"] == choice["domain"]], per_truth, choice["choice_id"])
    pools["U00"] = take([row for row in rows if row["derived_class_label"] == "unsupported"], per_truth, "U00")
    fixtures = []
    for truth in [row["choice_id"] for row in catalog["choices"] if row["choice_id"] != "A00"]:
        for candidate_index, candidate in enumerate(candidates):
            for row in pools[truth][candidate_index * spec["recordsPerTruthCandidateCell"]:(candidate_index + 1) * spec["recordsPerTruthCandidateCell"]]:
                fixtures.append({"fixture_id": f"v134::{truth}::{candidate}::{row['candidate_id']}", "truth_choice_id": truth, "presented_candidate_choice_id": candidate, "observation_available": True, **row})
    for candidate in candidates:
        for index in range(spec["recordsPerTruthCandidateCell"]): fixtures.append({"fixture_id": f"v134::A00::{candidate}::missing::{index:02d}", "truth_choice_id": "A00", "presented_candidate_choice_id": candidate, "observation_available": False, "candidate_id": None, "partition": None, "domain": None, "service": None, "intent": None, "derived_class_label": "insufficient"})
    fixtures.sort(key=lambda row: row["fixture_id"])
    cell_counts = Counter((row["truth_choice_id"], row["presented_candidate_choice_id"]) for row in fixtures); truth_counts = Counter(row["truth_choice_id"] for row in fixtures); candidate_counts = Counter(row["presented_candidate_choice_id"] for row in fixtures); source_ids = [row["candidate_id"] for row in fixtures if row["observation_available"]]
    return {"fixture_count": len(fixtures), "source_record_count": len(source_ids), "missing_control_count": sum(not row["observation_available"] for row in fixtures), "cell_count": len(cell_counts), "cell_counts": {f"{t}::{c}": n for (t,c),n in sorted(cell_counts.items())}, "truth_choice_counts": dict(sorted(truth_counts.items())), "presented_candidate_counts": dict(sorted(candidate_counts.items())), "known_pair_coverage": len({f"{row['service']}::{row['intent']}" for row in fixtures if row["derived_class_label"] == "known"}), "novel_domain_coverage": len({row["domain"] for row in fixtures if row["derived_class_label"] == "novel_valid"}), "unsupported_domain_coverage": len({row["domain"] for row in fixtures if row["derived_class_label"] == "unsupported"}), "unique_source_identifier_count": len(set(source_ids)), "contains_language": False, "fixtures_sha256": canonical_sha256(fixtures), "fixtures": fixtures}


def schema_identifiability(archive_bytes: bytes, catalog: dict[str, Any], population: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    shim = {"schemaAudit": {"archiveRevision": "e852981ae34990f4358979625854259302feaa78", "schemaPartitions": config["schemaIdentifiability"]["schemaPartitions"]}}
    schemas = read_schema_definitions(archive_bytes, shim); known = [schemas["train"][row["intent_id"]] for row in catalog["choices"] if row["kind"] == "KNOWN"]
    selected = Counter(f"{row['service']}::{row['intent']}" for row in population["fixtures"] if row["derived_class_label"] == "novel_valid")
    rows = []
    for intent_id, count in sorted(selected.items()):
        novel = schemas["dev"][intent_id]; relations = [compare_definitions(novel, item) for item in known]
        rows.append({"intent_id": intent_id, "selected_record_count": count, "exact_name_collision": any(row["exact_name"] for row in relations), "exact_full_signature_collision": any(row["exact_full_signature"] for row in relations)})
    total = sum(row["selected_record_count"] for row in rows)
    name_fraction = sum(row["selected_record_count"] for row in rows if row["exact_name_collision"]) / total; full_fraction = sum(row["selected_record_count"] for row in rows if row["exact_full_signature_collision"]) / total
    by_choice = {choice["choice_id"]: [row for row in rows if row["intent_id"] in choice["member_intent_ids"]] for choice in catalog["choices"] if choice["kind"] == "NOVEL_COMPOSITE"}
    return {"selected_novel_record_count": total, "selected_novel_exact_name_collision_fraction": name_fraction, "selected_novel_full_signature_collision_fraction": full_fraction, "entirely_name_colliding_novel_choice_count": sum(bool(group) and all(row["exact_name_collision"] for row in group) for group in by_choice.values()), "pair_summary": rows, "manual_semantic_judgment_count": 0, "raw_description_emission_count": 0}


def evaluate_gates(catalog: dict[str, Any], population: dict[str, Any], ident: dict[str, Any], config: dict[str, Any]) -> dict[str, bool]:
    gates = config["sourceGates"]; kinds = Counter(row["kind"] for row in catalog["choices"]); schema = config["schemaIdentifiability"]
    return {"choice_count": catalog["choice_count"] == gates["requiredChoiceCount"], "known_choice_count": kinds["KNOWN"] == gates["requiredKnownChoiceCount"], "novel_choice_count": kinds["NOVEL_COMPOSITE"] == gates["requiredNovelChoiceCount"], "unsupported_choice_count": kinds["UNSUPPORTED_COMPOSITE"] == gates["requiredUnsupportedChoiceCount"], "insufficient_choice_count": kinds["ABSTAIN"] == gates["requiredInsufficientChoiceCount"], "cell_count": population["cell_count"] == gates["requiredCellCount"], "records_per_cell": set(population["cell_counts"].values()) == {gates["requiredRecordsPerCell"]}, "records_per_truth": set(population["truth_choice_counts"].values()) == {gates["requiredRecordsPerTruth"]}, "records_per_candidate": set(population["presented_candidate_counts"].values()) == {gates["requiredRecordsPerCandidate"]}, "source_record_count": population["source_record_count"] == gates["requiredSourceRecordCount"], "missing_control_count": population["missing_control_count"] == gates["requiredMissingControlCount"], "fixture_count": population["fixture_count"] == gates["requiredFixtureCount"], "known_pair_coverage": population["known_pair_coverage"] == gates["requiredKnownPairCoverage"], "novel_domain_coverage": population["novel_domain_coverage"] == gates["requiredNovelDomainCoverage"], "unsupported_domain_coverage": population["unsupported_domain_coverage"] == gates["requiredUnsupportedDomainCoverage"], "unique_source_identifiers": population["unique_source_identifier_count"] == population["source_record_count"], "zero_name_collision": ident["selected_novel_exact_name_collision_fraction"] == schema["requiredSelectedNovelExactNameCollisionFraction"], "zero_full_signature_collision": ident["selected_novel_full_signature_collision_fraction"] == schema["requiredSelectedNovelFullSignatureCollisionFraction"], "zero_entirely_colliding_novel_choice": ident["entirely_name_colliding_novel_choice_count"] == schema["requiredEntirelyNameCollidingNovelChoiceCount"], "text_free_zero_model_execution": not catalog["contains_language"] and not population["contains_language"] and gates["maximumLanguageReadCount"] == gates["maximumManualLanguageOrRawResponseInspectionCount"] == gates["maximumModelLoadCount"] == gates["maximumModelGenerationCount"] == gates["maximumActualExecutionCount"] == 0}


__all__ = ["build_catalog", "derive_classes", "evaluate_gates", "schema_identifiability", "select_population"]
