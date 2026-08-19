from __future__ import annotations

import unittest

from v189_multiway_typed_channel_feasibility import MultiwayQuestion, evaluate_sequence


class V190FixedPolicyTest(unittest.TestCase):
    def test_domain_then_intent_stops_early(self) -> None:
        ids = ("a", "b", "c")
        problem = {
            "contract_ids": ids,
            "index": {key: i for i, key in enumerate(ids)},
            "prior": {key: 1 / 3 for key in ids},
            "generic_cost": 0.4,
            "questions": (
                MultiwayQuestion("M189_domain", "domain", ("x", "y", "y")),
                MultiwayQuestion("M189_intent_concept", "intent", ("a", "b", "c")),
            ),
        }
        scenario = {"rule": "bit_slot", "turn_overhead": 0.0}
        early = evaluate_sequence(problem, scenario, ("M189_domain", "M189_intent_concept"), "a")
        late = evaluate_sequence(problem, scenario, ("M189_domain", "M189_intent_concept"), "b")
        self.assertEqual(early["turn_count"], 1)
        self.assertEqual(late["turn_count"], 2)
        self.assertTrue(early["final_exact"] and late["final_exact"])


if __name__ == "__main__":
    unittest.main()
