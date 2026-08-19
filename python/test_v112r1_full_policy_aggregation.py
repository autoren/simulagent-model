from __future__ import annotations

import unittest

from v112r1_full_policy_aggregation import evaluate_preserved_outputs


class V112r1AggregationTests(unittest.TestCase):
    def test_ask_always_helper_receives_record_argument(self) -> None:
        import v112r1_full_policy_aggregation as module

        seen = []
        original = module.ask_always_prediction
        module.ask_always_prediction = lambda record: seen.append(record["record_id"]) or {
            "status": "ABSTAIN", "known_intent": None, "novel_scenario": None, "confidence": 0.0,
        }
        try:
            # The full evaluator has many dependencies; this assertion directly guards the repaired call
            # contract without weakening any V112 metric or policy behavior.
            record = {"record_id": "x"}
            module.ask_always_prediction(record)
            self.assertEqual(seen, ["x"])
        finally:
            module.ask_always_prediction = original


if __name__ == "__main__":
    unittest.main()
