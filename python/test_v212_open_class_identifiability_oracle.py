from __future__ import annotations

import json
import unittest
from pathlib import Path

from v212_open_class_identifiability_oracle import (
    behavior_id,
    classify_behavior,
    equivalent_rewrite,
    evaluate_expression,
    language_catalog,
    materialize_cases,
    materialize_public_semantics,
    rename_record,
    resolve_episode,
    reverse_commutative_order,
    reverse_evidence_order,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = json.loads((ROOT / "configs/v212-open-class-identifiability-oracle.json").read_text())
SEMANTICS = materialize_public_semantics(CONFIG)


class V212OracleTests(unittest.TestCase):
    def test_finite_algebra_and_representation_precedence(self) -> None:
        primitives = SEMANTICS["registered_primitives"]
        conjunction = {"op": "AND", "args": [{"op": "PRIMITIVE", "name": "P"}, {"op": "PRIMITIVE", "name": "Q"}]}
        exclusive_or = {"op": "XOR", "args": [{"op": "PRIMITIVE", "name": "P"}, {"op": "PRIMITIVE", "name": "Q"}]}
        self.assertEqual("00000011", evaluate_expression(conjunction, primitives))
        catalog = language_catalog(SEMANTICS)
        self.assertEqual("EXISTING_PRIMITIVE", classify_behavior(behavior_id(primitives["P"]), catalog))
        self.assertEqual("EXISTING_COMPOSITION", classify_behavior(behavior_id(evaluate_expression(conjunction, primitives)), catalog))
        self.assertEqual("MISSING_OPERATOR", classify_behavior(behavior_id(evaluate_expression(exclusive_or, primitives)), catalog))
        self.assertEqual("IRREDUCIBLE_PROVISIONAL", classify_behavior(behavior_id("00010111"), catalog))

    def test_public_materialization_has_no_hidden_truth_fields(self) -> None:
        public, truth = materialize_cases(CONFIG, SEMANTICS)
        self.assertEqual(40, len(public))
        self.assertEqual(40, len(truth))
        hidden = set(CONFIG["population"]["hiddenFields"]) - {"case_id"}
        self.assertTrue(all(not (set(row) & hidden) for row in public))
        self.assertEqual({row["case_id"] for row in public}, {row["case_id"] for row in truth})

    def test_synthetic_ambiguity_and_contradiction(self) -> None:
        base = {
            "case_id": "synthetic",
            "definition": {"kind": "UNCONSTRAINED"},
            "references": [],
            "reference_facts": [],
            "observations": [{"world": world, "output": int(bit)} for world, bit in zip(SEMANTICS["world_order"][:-1], "0000111")],
            "comparison_anchor": None,
        }
        ambiguous = resolve_episode(base, SEMANTICS)
        self.assertEqual("AMBIGUOUS", ambiguous["evidence_status"])
        self.assertEqual(2, len(ambiguous["candidate_ids"]))
        conflict = {**base, "observations": [{"world": "000", "output": 0}, {"world": "000", "output": 1}]}
        self.assertEqual("CONTRADICTORY", resolve_episode(conflict, SEMANTICS)["evidence_status"])

    def test_public_invariance_transforms_preserve_resolution(self) -> None:
        public, _ = materialize_cases(CONFIG, SEMANTICS)
        examples = [
            next(row for row in public if row["definition"]["kind"] == "SYMBOL"),
            next(row for row in public if row["definition"].get("expression", {}).get("op") == "AND"),
        ]
        for row in examples:
            expected = resolve_episode(row, SEMANTICS)["candidate_ids"]
            for transform in (rename_record, reverse_evidence_order, reverse_commutative_order, equivalent_rewrite):
                self.assertEqual(expected, resolve_episode(transform(row), SEMANTICS)["candidate_ids"])


if __name__ == "__main__":
    unittest.main()
