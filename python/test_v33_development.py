import json
import unittest
from pathlib import Path

import numpy as np

from v33_development import (
    combine_outputs, qualification_checks, select_qualified_system,
    select_search_checkpoint,
)


class V33DevelopmentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = json.loads(Path("configs/v33-development-adequacy.json").read_text())

    def test_modular_output_assembly_uses_registered_sources(self):
        modules = []
        for base in (10, 20, 30, 40):
            modules.append(tuple(np.full((2, index + 1), base + index) for index in range(6)))
        result = combine_outputs(*modules)
        self.assertEqual([int(value[0, 0]) for value in result], [10, 11, 12, 23, 34, 45])

    def test_search_selection_prefers_calibration_then_fewer_epochs(self):
        base = {
            "atom_exact_accuracy": 0.9, "lexical_sign_accuracy": 0.9,
            "outer_operation_accuracy": 0.9, "compiled_truth_accuracy": 0.9,
        }
        reports = [
            {"learning_rate": 0.001, "epoch": 4, "fit": dict(base), "calibration": dict(base)},
            {"learning_rate": 0.003, "epoch": 2, "fit": dict(base), "calibration": dict(base)},
        ]
        chosen = select_search_checkpoint("atom", reports, self.config)
        self.assertEqual(chosen["epoch"], 2)

    def test_qualification_is_noncompensatory(self):
        fit = {"atom_exact_accuracy": 1.0, "lexical_sign_accuracy": 1.0, "outer_operation_accuracy": 1.0, "compiled_truth_accuracy": 1.0, "compiled_exact_fact_accuracy": 1.0}
        calibration = {**fit, "relation_order_accuracy": 0.89}
        checks = qualification_checks(fit, calibration, self.config)
        self.assertFalse(checks["calibration_relation_order"])
        self.assertFalse(all(checks.values()))

    def test_selection_prefers_joint_without_material_modular_advantage(self):
        systems = {
            "jointCompiled": {"passed": True, "calibration_mean": {"compiled_exact_fact_accuracy": 0.92}},
            "independentCompiled": {"passed": True, "calibration_mean": {"compiled_exact_fact_accuracy": 0.94}},
        }
        selected, _ = select_qualified_system(systems, self.config)
        self.assertEqual(selected, "jointCompiled")


if __name__ == "__main__": unittest.main()
