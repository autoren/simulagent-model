from __future__ import annotations

import json
import unittest

from v22r2_grounding import PROJECT_ROOT
from run_v169_fresh_constraint_state_population import reconstruct
from v169r1_json_key_normalization_repair import json_normalize, only_class_coverage_key_type_mismatch


class V169r1JSONKeyNormalizationRepairTest(unittest.TestCase):
    def test_sole_mismatch_normalizes_exactly(self) -> None:
        lock = json.loads((PROJECT_ROOT / "configs/v169-fresh-constraint-state-population-lock.json").read_text())
        result = json.loads((PROJECT_ROOT / "outputs/v169-fresh-constraint-state-population/population/result.json").read_text())
        reconstructed = reconstruct(lock)["population"]["summary"]
        self.assertTrue(only_class_coverage_key_type_mismatch(result["summary"], reconstructed))
        self.assertEqual(result["summary"], json_normalize(reconstructed))


if __name__ == "__main__":
    unittest.main()
