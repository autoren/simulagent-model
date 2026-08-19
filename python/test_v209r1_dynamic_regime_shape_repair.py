from __future__ import annotations

import json
import unittest
from pathlib import Path

from v209r1_dynamic_regime_shape_repair import repair_diagnostics


CONFIG = json.loads(Path("configs/v209-controlled-language-observation-pomdp-lock.json").read_text())["config_payload"]


class V209r1RepairTest(unittest.TestCase):
    def test_one_two_and_three_regime_kernels_construct_with_fixed_other_axes(self) -> None:
        diagnostics = repair_diagnostics(CONFIG)
        self.assertTrue(diagnostics["one_two_three_regime_kernels_construct"])
        for count in (1, 2, 3):
            row = diagnostics["regime_shapes"][str(count)]
            self.assertEqual(row["reference"], [count, 2, 3])
            self.assertEqual(row["target"], [count, 2, 3])
            self.assertEqual(row["ask_reference_cost"], [count, 2])
            self.assertEqual(row["ask_target_cost"], [count, 2])

    def test_repair_changes_no_scientific_design_counts(self) -> None:
        diagnostics = repair_diagnostics(CONFIG)
        self.assertTrue(diagnostics["parent_config_used_without_copy"])
        self.assertEqual(diagnostics["changed_scientific_parameter_count"], 0)
        self.assertEqual(diagnostics["changed_gate_count"], 0)
        self.assertEqual(diagnostics["changed_comparator_count"], 0)
        self.assertEqual(diagnostics["changed_decision_rule_count"], 0)


if __name__ == "__main__":
    unittest.main()
