from __future__ import annotations

from fractions import Fraction
import unittest

from v188_binary_clarification_channel_frontier import information_controls, restricted_exact_depth_tree


class V188FrontierTest(unittest.TestCase):
    def test_uniform_four_leaf_information_controls(self) -> None:
        problem = {
            "contract_ids": ("a", "b", "c", "d"),
            "prior": {key: Fraction(1, 4) for key in ("a", "b", "c", "d")},
        }
        result = information_controls(problem)
        self.assertEqual(result["shannon_entropy_bits"], 2.0)
        self.assertEqual(result["huffman_expected_depth"], 2.0)
        self.assertTrue(result["entropy_lower_bound_pass"])

    def test_restricted_two_bit_tree(self) -> None:
        from v187_clean_typed_clarification_planner import Question
        ids = ("a", "b", "c", "d")
        problem = {
            "contract_ids": ids,
            "contract_index": {key: i for i, key in enumerate(ids)},
            "prior": {key: Fraction(1, 4) for key in ids},
            "questions": (
                Question("q0", "bit", 0, (0, 0, 1, 1)),
                Question("q1", "bit", 1, (0, 1, 0, 1)),
            ),
        }
        result = restricted_exact_depth_tree(problem, 3)
        self.assertEqual(result["expected_depth"], 2.0)
        self.assertEqual(result["leaf_count"], 4)
        self.assertEqual(result["maximum_depth"], 2)


if __name__ == "__main__":
    unittest.main()
