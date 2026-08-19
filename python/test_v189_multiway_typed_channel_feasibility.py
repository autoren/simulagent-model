from __future__ import annotations

import unittest

from v189_multiway_typed_channel_feasibility import MultiwayQuestion, question_cost


class V189MultiwayTest(unittest.TestCase):
    def test_bit_slot_anchor_and_four_way_cost(self) -> None:
        problem = {
            "contract_ids": ("a", "b", "c", "d"),
            "index": {"a": 0, "b": 1, "c": 2, "d": 3},
            "prior": {key: 0.25 for key in ("a", "b", "c", "d")},
        }
        binary = MultiwayQuestion("b", "b", (0, 0, 1, 1))
        four = MultiwayQuestion("f", "f", (0, 1, 2, 3))
        scenario = {"rule": "bit_slot", "turn_overhead": 0.03}
        self.assertAlmostEqual(question_cost(problem, problem["contract_ids"], binary, scenario), 0.10)
        self.assertAlmostEqual(question_cost(problem, problem["contract_ids"], four, scenario), 0.17)

    def test_entropy_lower_bound_four_uniform(self) -> None:
        problem = {
            "contract_ids": ("a", "b", "c", "d"),
            "index": {"a": 0, "b": 1, "c": 2, "d": 3},
            "prior": {key: 0.25 for key in ("a", "b", "c", "d")},
        }
        four = MultiwayQuestion("f", "f", (0, 1, 2, 3))
        self.assertAlmostEqual(question_cost(problem, problem["contract_ids"], four, {"rule": "entropy_lower_bound", "turn_overhead": 0}), 0.2)


if __name__ == "__main__":
    unittest.main()
