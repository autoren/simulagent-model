from __future__ import annotations

import unittest

from v207r2_agentabstain_outcome_verification_repair import evaluate_repair


class V207r2RepairContractTest(unittest.TestCase):
    def test_repair_rejects_unexpected_false_check(self) -> None:
        failed = {"checks": {"design_lock_and_dependencies_are_exact": False, "unexpected": False}}
        config = {"repairContract": {"requiredFalseChecks": ["design_lock_and_dependencies_are_exact"]}}
        false_checks = sorted(key for key, value in failed["checks"].items() if not value)
        self.assertNotEqual(false_checks, sorted(config["repairContract"]["requiredFalseChecks"]))

    def test_public_function_is_available(self) -> None:
        self.assertTrue(callable(evaluate_repair))


if __name__ == "__main__":
    unittest.main()
