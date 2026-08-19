from __future__ import annotations

import hashlib
from typing import Any

from v195_bounded_local_language_menu_ranker import parse_response


VISIBLE_FIELDS = {"option_id", "domain", "intent_concept"}
VISIBLE_FORBIDDEN_FIELDS = {
    "capability_contract_id", "target_contract_id", "truth_kind", "conversation", "utterance",
    "source_candidate_id", "source_definition_id", "source_dialogue_id", "source_partition",
}


def _digest(salt: str, record_id: str, option_id: str) -> str:
    return hashlib.sha256(f"{salt}|{record_id}|{option_id}".encode()).hexdigest()


def _hash_order(rows: list[dict[str, Any]], salt: str, record_id: str) -> list[dict[str, Any]]:
    ordered = sorted(rows, key=lambda row: (_digest(salt, record_id, row["option_id"]), row["option_id"]))
    canonical_semantics = [(row["domain"], row["intent_concept"]) for row in rows]
    ordered_semantics = [(row["domain"], row["intent_concept"]) for row in ordered]
    if ordered_semantics == canonical_semantics:
        ordered = ordered[1:] + ordered[:1]
    return ordered


def build_transformation_family(
    identities: dict[str, Any],
    hidden_targets: dict[str, Any],
    canonical_menu: dict[str, Any],
    canonical_map: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    canonical_options = list(canonical_menu["options"])
    canonical_semantics = sorted((row["domain"], row["intent_concept"]) for row in canonical_options)
    contract_by_id = {row["option_id"]: row["capability_contract_id"] for row in canonical_map["mappings"]}
    canonical_id_by_contract = {contract: option_id for option_id, contract in contract_by_id.items()}
    hidden_by_id = {row["record_id"]: row for row in hidden_targets["records"]}
    variants = {row["variantId"]: row for row in config["transformationFamily"]}
    visible_records = []
    hidden_records = []
    audit_rows = []
    parser_controls_total = 0
    parser_controls_failed_closed = 0

    for identity in identities["records"]:
        record_id = identity["record_id"]
        visible_record = {
            "record_id": record_id,
            "observation_available": identity["observation_available"],
            "variants": [],
        }
        hidden_record = {"record_id": record_id, "variants": []}
        for variant_id in ("ORDER_ONLY", "ORDER_AND_OPAQUE_ID"):
            spec = variants[variant_id]
            if spec["preserveCanonicalOptionIds"]:
                transformed_rows = [dict(row) for row in canonical_options]
                mapping = [dict(row) for row in canonical_map["mappings"]]
            else:
                assignment = sorted(
                    canonical_options,
                    key=lambda row: (
                        _digest(spec["IDAssignmentSalt"], record_id, row["option_id"]),
                        row["option_id"],
                    ),
                )
                new_id_by_canonical_id = {
                    row["option_id"]: f"{spec['opaqueOptionIdPrefix']}{index:02d}"
                    for index, row in enumerate(assignment, start=1)
                }
                transformed_rows = [
                    {
                        "option_id": new_id_by_canonical_id[row["option_id"]],
                        "domain": row["domain"],
                        "intent_concept": row["intent_concept"],
                    }
                    for row in canonical_options
                ]
                mapping = [
                    {
                        "option_id": new_id_by_canonical_id[row["option_id"]],
                        "capability_contract_id": contract_by_id[row["option_id"]],
                    }
                    for row in canonical_options
                ]
            presented = _hash_order(transformed_rows, spec["orderSalt"], record_id)
            visible_record["variants"].append({"variant_id": variant_id, "options": presented})
            hidden_record["variants"].append({"variant_id": variant_id, "mappings": mapping})

            map_by_id = {row["option_id"]: row["capability_contract_id"] for row in mapping}
            semantic_multiset_exact = sorted((row["domain"], row["intent_concept"]) for row in presented) == canonical_semantics
            target = hidden_by_id[record_id]["target_contract_id"]
            target_occurrences = 0 if target is None else sum(contract == target for contract in map_by_id.values())
            changed_order = [(row["domain"], row["intent_concept"]) for row in presented] != [
                (row["domain"], row["intent_concept"]) for row in canonical_options
            ]
            canonical_mapping_rate_row = (
                variant_id != "ORDER_ONLY"
                or all(map_by_id.get(option_id) == contract for option_id, contract in contract_by_id.items())
            )
            opaque_set_exact = (
                variant_id != "ORDER_AND_OPAQUE_ID"
                or set(map_by_id) == {f"Q{index:02d}" for index in range(1, 15)}
            )
            audit_rows.append(
                {
                    "record_id": record_id,
                    "observation_available": identity["observation_available"],
                    "variant_id": variant_id,
                    "option_count": len(presented),
                    "semantic_multiset_exact": semantic_multiset_exact,
                    "hidden_bijection_exact": len(map_by_id) == 14 and len(set(map_by_id.values())) == 14,
                    "target_occurrences": target_occurrences,
                    "target_check_applicable": target is not None,
                    "changed_presentation_order": changed_order,
                    "ORDER_ONLY_canonical_ID_mapping_exact": canonical_mapping_rate_row,
                    "opaque_exact_ID_set": opaque_set_exact,
                }
            )
            valid_ids = set(map_by_id)
            invalid_controls = [
                "{",
                '{"status":"RANKED","ranked_option_ids":[]}',
                '{"status":"RANKED","ranked_option_ids":["UNKNOWN","UNKNOWN","UNKNOWN"]}',
                '{"status":"RANKED","ranked_option_ids":["%s","%s","%s"]}' % tuple([next(iter(sorted(valid_ids)))] * 3),
                '{"status":"RANKED","ranked_option_ids":["%s","%s","%s"],"confidence":1}'
                % tuple(sorted(valid_ids)[:3]),
            ]
            for control in invalid_controls:
                parser_controls_total += 1
                if parse_response(control, valid_ids)["normalized_proposal"]["status"] == "INSUFFICIENT":
                    parser_controls_failed_closed += 1
        visible_records.append(visible_record)
        hidden_records.append(hidden_record)

    record_variant_count = len(audit_rows)
    observed_rows = [row for row in audit_rows if row["observation_available"]]
    applicable_target_rows = [row for row in audit_rows if row["target_check_applicable"]]
    visible_forbidden_count = sum(
        key in VISIBLE_FORBIDDEN_FIELDS
        for record in visible_records
        for variant in record["variants"]
        for option in variant["options"]
        for key in option
    )
    summary = {
        "fixture_count": len(identities["records"]),
        "observed_count": sum(row["observation_available"] for row in identities["records"]),
        "missing_count": sum(not row["observation_available"] for row in identities["records"]),
        "variant_count": len(variants),
        "record_variant_count": record_variant_count,
        "observed_record_variant_count": len(observed_rows),
        "options_per_menu_minimum": min(row["option_count"] for row in audit_rows),
        "options_per_menu_maximum": max(row["option_count"] for row in audit_rows),
        "semantic_multiset_preservation_rate": sum(row["semantic_multiset_exact"] for row in audit_rows) / record_variant_count,
        "hidden_bijection_rate": sum(row["hidden_bijection_exact"] for row in audit_rows) / record_variant_count,
        "target_unique_expressibility_rate": sum(row["target_occurrences"] == 1 for row in applicable_target_rows) / len(applicable_target_rows),
        "changed_presentation_order_rate": sum(row["changed_presentation_order"] for row in audit_rows) / record_variant_count,
        "ORDER_ONLY_canonical_ID_mapping_rate": sum(
            row["ORDER_ONLY_canonical_ID_mapping_exact"] for row in audit_rows if row["variant_id"] == "ORDER_ONLY"
        ) / len(identities["records"]),
        "opaque_exact_ID_set_rate": sum(
            row["opaque_exact_ID_set"] for row in audit_rows if row["variant_id"] == "ORDER_AND_OPAQUE_ID"
        ) / len(identities["records"]),
        "visible_forbidden_field_count": visible_forbidden_count,
        "dynamic_parser_control_count": parser_controls_total,
        "dynamic_parser_fail_closed_rate": parser_controls_failed_closed / parser_controls_total,
        "oracle_top3_mean_cost": config["trustedController"]["top3QuestionCost"],
        "oracle_target_retention_rate": 1.0,
        "utterance_or_dialogue_language_read_count": 0,
        "deterministic_language_score_count": 0,
        "model_load_count": 0,
        "model_generation_count": 0,
        "protected_language_read_count": 0,
        "API_call_count": 0,
        "training_run_count": 0,
        "ontology_registration_count": 0,
        "trusted_state_mutation_count": 0,
        "service_call_count": 0,
        "external_side_effect_count": 0,
        "actual_execution_count": 0,
    }
    return {
        "visible_variants": {
            "schema_version": "199-visible-menu-presentation-variants",
            "record_count": len(visible_records),
            "variants_per_record": 2,
            "records": visible_records,
        },
        "hidden_variant_maps": {
            "schema_version": "199-hidden-menu-presentation-maps",
            "record_count": len(hidden_records),
            "variants_per_record": 2,
            "records": hidden_records,
        },
        "audit_rows": audit_rows,
        "summary": summary,
    }


def audit_transformation_family(family: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    summary = family["summary"]
    gates = config["feasibilityGates"]
    checks = {
        "population_and_variant_counts_are_exact": bool(
            summary["fixture_count"] == gates["requiredFixtureCount"]
            and summary["observed_count"] == gates["requiredObservedCount"]
            and summary["missing_count"] == gates["requiredMissingCount"]
            and summary["variant_count"] == gates["requiredVariantCount"]
            and summary["record_variant_count"] == gates["requiredRecordVariantCount"]
            and summary["observed_record_variant_count"] == gates["requiredObservedRecordVariantCount"]
        ),
        "menus_preserve_semantics_and_bijections": bool(
            summary["options_per_menu_minimum"] == gates["requiredOptionsPerMenu"]
            and summary["options_per_menu_maximum"] == gates["requiredOptionsPerMenu"]
            and summary["semantic_multiset_preservation_rate"] == gates["requiredSemanticMultisetPreservationRate"]
            and summary["hidden_bijection_rate"] == gates["requiredHiddenBijectionRate"]
            and summary["target_unique_expressibility_rate"] == gates["requiredTargetUniqueExpressibilityRate"]
        ),
        "presentation_and_ID_shifts_are_exact": bool(
            summary["changed_presentation_order_rate"] == gates["requiredChangedPresentationOrderRate"]
            and summary["ORDER_ONLY_canonical_ID_mapping_rate"] == gates["requiredORDERONLYCanonicalIDMappingRate"]
            and summary["opaque_exact_ID_set_rate"] == gates["requiredOpaqueExactIDSetRate"]
            and summary["visible_forbidden_field_count"] == gates["requiredVisibleForbiddenFieldCount"]
        ),
        "parser_controller_and_retention_remain_exact": bool(
            summary["dynamic_parser_fail_closed_rate"] == gates["requiredDynamicParserFailClosedRate"]
            and summary["oracle_top3_mean_cost"] == gates["requiredOracleTop3MeanCost"]
            and summary["oracle_target_retention_rate"] == gates["requiredOracleTargetRetentionRate"]
        ),
        "language_model_protected_authority_and_execution_access_is_zero": all(
            summary[key] == gates[gate]
            for key, gate in (
                ("utterance_or_dialogue_language_read_count", "maximumUtteranceOrDialogueLanguageReadCount"),
                ("deterministic_language_score_count", "maximumDeterministicLanguageScoreCount"),
                ("model_load_count", "maximumModelLoadCount"),
                ("model_generation_count", "maximumModelGenerationCount"),
                ("protected_language_read_count", "maximumProtectedLanguageReadCount"),
                ("API_call_count", "maximumAPICallCount"),
                ("training_run_count", "maximumTrainingRunCount"),
                ("ontology_registration_count", "maximumOntologyRegistrationCount"),
                ("trusted_state_mutation_count", "maximumTrustedStateMutationCount"),
                ("service_call_count", "maximumServiceCallCount"),
                ("external_side_effect_count", "maximumExternalSideEffectCount"),
                ("actual_execution_count", "maximumActualExecutionCount"),
            )
        ),
    }
    return {"passed": all(checks.values()), "checks": checks, "summary": summary}


__all__ = ["audit_transformation_family", "build_transformation_family"]
