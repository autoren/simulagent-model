from __future__ import annotations

import unittest

from v214_deterministic_candidate_version_space_controls import (
    _canonical_expression,
    _jaccard_distance,
    normalized_signature,
)


class V214ControlTests(unittest.TestCase):
    def test_identity_and_commutative_order_canonicalize(self) -> None:
        left = {"op": "IDENTITY", "arg": {"op": "AND", "args": [{"op": "PRIMITIVE", "name": "P"}, {"op": "PRIMITIVE", "name": "Q"}]}}
        right = {"op": "AND", "args": [{"op": "PRIMITIVE", "name": "Q"}, {"op": "PRIMITIVE", "name": "P"}]}
        self.assertEqual(_canonical_expression(left), _canonical_expression(right))

    def test_typed_jaccard_distance(self) -> None:
        self.assertEqual(0.0, _jaccard_distance({"a", "b"}, {"a", "b"}))
        self.assertEqual(1.0, _jaccard_distance({"a"}, {"b"}))

    def test_normalization_excludes_case_group_split_and_variant(self) -> None:
        base = {
            "case_id": "case-a",
            "group_id": "group-a",
            "split": "development",
            "variant_code": "V0",
            "definition": {"kind": "EXPRESSION", "expression": {"op": "PRIMITIVE", "name": "P"}},
            "references": [],
            "reference_facts": [],
            "observations": [],
            "comparison_anchor": None,
        }
        changed = {**base, "case_id": "case-b", "group_id": "group-b", "variant_code": "V3"}
        self.assertEqual(normalized_signature(base), normalized_signature(changed))


if __name__ == "__main__":
    unittest.main()
