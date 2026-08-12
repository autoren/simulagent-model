import json
import unittest
from pathlib import Path

from v9_symbolic import evaluate_allowed_transitions


class V9SymbolicTests(unittest.TestCase):
    def test_python_evaluator_reproduces_every_v9r2_target(self):
        records = []
        for split in ("train", "calibration"):
            records.extend(
                json.loads(line)
                for line in Path(f"data/v9r2/records/{split}.jsonl").read_text().splitlines()
                if line
            )
        self.assertEqual(len(records), 2160)
        for record in records:
            result = evaluate_allowed_transitions(
                record["action_dependency_schema"],
                record["target"]["determinant_grounding"],
            )
            self.assertEqual(result["identifiable"], record["target"]["identifiable"])
            self.assertEqual(
                result["possible_transition_codes"],
                record["target"]["possible_transition_codes"],
            )

    def test_rejects_missing_determinant(self):
        schema = {
            "transition_determinants": [{"id": "a"}],
            "transition_cases": [
                {"values": ["inactive"], "transition_code": "x"},
                {"values": ["active"], "transition_code": "y"},
            ],
        }
        with self.assertRaisesRegex(ValueError, "omits determinant"):
            evaluate_allowed_transitions(schema, [])


if __name__ == "__main__":
    unittest.main()
