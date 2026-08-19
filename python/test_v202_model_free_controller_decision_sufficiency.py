from __future__ import annotations

import unittest

from v202_model_free_controller_decision_sufficiency import (
    _consensus,
    _cost,
    _plurality,
)


CONFIG = {
    "trustedController": {
        "bitCost": 0.10,
        "targetOutsideProposalAdditionalGenericCost": 0.40,
    }
}


class V202ControllerTest(unittest.TestCase):
    def test_frozen_question_costs_and_miss_penalty(self) -> None:
        self.assertEqual(_cost([], "A", 0.3, CONFIG), 0.3)
        self.assertEqual(_cost(["A"], "A", 0.3, CONFIG), 0.1)
        self.assertEqual(_cost(["B"], "A", 0.3, CONFIG), 0.5)
        self.assertEqual(_cost(["A", "B"], "A", 0.3, CONFIG), 0.2)
        self.assertEqual(_cost(["A", "B", "C"], "A", 0.3, CONFIG), 0.2)
        self.assertAlmostEqual(_cost(["A", "B", "C", "D"], "A", 0.3, CONFIG), 0.3)

    def test_plurality_uses_canonical_top1_for_three_way_tie(self) -> None:
        sources = {
            "CANONICAL": ["B"],
            "ORDER_ONLY": ["A"],
            "ORDER_AND_OPAQUE_ID": ["C"],
        }
        self.assertEqual(_plurality(sources, list(sources)), ["B"])

    def test_plurality_uses_majority_before_canonical(self) -> None:
        sources = {
            "CANONICAL": ["B"],
            "ORDER_ONLY": ["A"],
            "ORDER_AND_OPAQUE_ID": ["A"],
        }
        self.assertEqual(_plurality(sources, list(sources)), ["A"])

    def test_consensus_requires_two_presentations_and_uses_fixed_order(self) -> None:
        sources = {
            "CANONICAL": ["A", "B", "C"],
            "ORDER_ONLY": ["D", "B", "A"],
            "ORDER_AND_OPAQUE_ID": ["D", "A", "E"],
        }
        self.assertEqual(_consensus(sources, list(sources), 2, 4), ["A", "B", "D"])


if __name__ == "__main__":
    unittest.main()
