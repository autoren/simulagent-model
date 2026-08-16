#!/usr/bin/env python3
"""Post-hoc diagnosis of the failed V43 ordered-list graph comparison.

This diagnostic does not alter the sealed result or its qualification.
"""

from __future__ import annotations

from collections import Counter
import json

from v10_protocol import file_sha256
from v22_relational import canonical_json
from v22r2_grounding import PROJECT_ROOT
from v43_language import compile_state


def read(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def canonical_rows(rows):
    return sorted(rows, key=canonical_json)


def main():
    seal_path = PROJECT_ROOT / "configs/v43-corpus-seal.json"
    result_path = PROJECT_ROOT / "outputs/v43-sequential-language-grounding/development/result.json"
    output = PROJECT_ROOT / "outputs/v43-sequential-language-grounding/post-hoc-graph-diagnostic.json"
    markdown = PROJECT_ROOT / "docs/v43-post-hoc-graph-diagnostic.md"
    seal = json.loads(seal_path.read_text())
    snapshots = []
    artifact_hashes = {}
    for artifact in seal["corpora"].values():
        path = PROJECT_ROOT / artifact["path"]
        artifact_hashes[artifact["path"]] = file_sha256(path)
        if artifact_hashes[artifact["path"]] != artifact["sha256"]:
            raise RuntimeError("V43 sealed corpus changed before diagnosis")
        for record in read(path):
            agent = record["agent_input"]
            support_refs = {row["id"]: row for row in record["reference"]["support_sequences"]}
            query_refs = {row["id"]: row for row in record["reference"]["queries"]}
            for public in agent["support_sequences"]:
                reference = support_refs[public["id"]]
                snapshots.append((record["id"], "support_initial", public["id"], public["entities"], public["initial_state_language"], reference["initial_state"], agent))
                for index, (language, expected) in enumerate(zip(public["observed_step_state_language"], reference["observed_step_states"]), start=1):
                    snapshots.append((record["id"], "support_step", f"{public['id']}:{index}", public["entities"], language, expected, agent))
            for public in agent["queries"]:
                reference = query_refs[public["id"]]
                snapshots.append((record["id"], "query_initial", public["id"], public["entities"], public["initial_state_language"], reference["initial_state"], agent))
    rows, first_order_only = [], None
    for record_id, kind, snapshot_id, entities, language, reference, agent in snapshots:
        compiled = compile_state(language, entities, agent["predicate_ontology"], agent["operator_ontology"])
        actual = compiled["epistemic_state"]
        expected = reference["epistemic_state"]
        ordered_equal = actual == expected
        canonical_equal = canonical_rows(actual) == canonical_rows(expected)
        actual_keys = [canonical_json(row) for row in actual]
        expected_keys = [canonical_json(row) for row in expected]
        duplicate_free = len(actual_keys) == len(set(actual_keys)) and len(expected_keys) == len(set(expected_keys))
        row = {
            "record": record_id,
            "kind": kind,
            "snapshot": snapshot_id,
            "compiler_status": compiled["status"],
            "ordered_equal": ordered_equal,
            "canonical_row_set_equal": canonical_equal,
            "duplicate_free": duplicate_free,
        }
        rows.append(row)
        if first_order_only is None and not ordered_equal and canonical_equal:
            first_order_only = {
                "record": record_id,
                "kind": kind,
                "snapshot": snapshot_id,
                "actual_atom_order": [item["atom"] for item in actual],
                "reference_atom_order": [item["atom"] for item in expected],
            }
    counts = Counter(row["kind"] for row in rows)
    diagnostic = {
        "schema_version": "43-post-hoc-1",
        "experiment": "v43_graph_metric_diagnostic",
        "status": "post_hoc_diagnostic_not_registered_result_revision",
        "sealed_result": str(result_path.relative_to(PROJECT_ROOT)),
        "sealed_result_sha256": file_sha256(result_path),
        "corpus_seal": str(seal_path.relative_to(PROJECT_ROOT)),
        "corpus_seal_sha256": file_sha256(seal_path),
        "sealed_artifact_hashes": artifact_hashes,
        "snapshots": len(rows),
        "snapshot_kinds": dict(counts),
        "ordered_list_exact": sum(row["ordered_equal"] for row in rows) / len(rows),
        "canonical_row_set_exact": sum(row["canonical_row_set_equal"] for row in rows) / len(rows),
        "compiler_status_ok": sum(row["compiler_status"] == "ok" for row in rows) / len(rows),
        "duplicate_free": sum(row["duplicate_free"] for row in rows) / len(rows),
        "semantic_content_mismatches": sum(not row["canonical_row_set_equal"] for row in rows),
        "ordering_only_mismatches": sum(not row["ordered_equal"] and row["canonical_row_set_equal"] for row in rows),
        "first_ordering_only_example": first_order_only,
        "diagnosis": "reference_rows_were_sorted_before_hashed_entity_aliasing_while_compiled_rows_were_sorted_after_aliasing" if all(row["canonical_row_set_equal"] for row in rows) else "semantic_mismatch_present",
        "scientific_constraint": "V43_remains_a_registered_gate_failure_until_a_separately_preregistered_measurement_repair_confirms_the_canonical_metric",
        "data_access": {
            "sealed_v43_records_read": 40,
            "new_development_runs": 0,
            "model_forward_passes": 0,
            "adapter_training_runs": 0,
        },
    }
    output.write_text(json.dumps(diagnostic, indent=2, sort_keys=True) + "\n")
    markdown.write_text("\n".join([
        "# V43 post-hoc graph-metric diagnostic",
        "",
        "This is a labeled post-hoc diagnostic. It does not revise V43's sealed failed qualification.",
        "",
        f"- Ordered-list exactness reproduced at `{diagnostic['ordered_list_exact']:.3f}`.",
        f"- Canonical row-set exactness was `{diagnostic['canonical_row_set_exact']:.3f}`.",
        f"- Semantic content mismatches: `{diagnostic['semantic_content_mismatches']}` of `{diagnostic['snapshots']}` graphs.",
        f"- Ordering-only mismatches: `{diagnostic['ordering_only_mismatches']}`.",
        "",
        "The reference rows were ordered using canonical entity IDs before those IDs were replaced with hashed aliases. Compiled rows were ordered after aliasing. Direct list equality therefore treated semantically identical graphs with different row order as unequal.",
        "",
        "The proper next step is a separately preregistered measurement-repair confirmation over the immutable V43 artifacts. Only the graph comparator may change; every other V43 metric must reproduce exactly.",
    ]) + "\n")
    print(json.dumps(diagnostic, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
