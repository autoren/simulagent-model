import unittest

from replay_v16_scope_correct_gates import scope_correct_report


class V16ScopeCorrectTests(unittest.TestCase):
    def test_uses_only_folds_with_complete_in_mask_groups(self):
        checks = [{
            "name": "minimum_fold_span_accuracy", "value": 0.8,
            "minimum": 0.65, "passed": True,
        }, {
            "name": "minimum_fold_complete_intervention_group_accuracy", "value": 0.1,
            "minimum": 0.5, "passed": False,
        }]
        folds = {"context": {}}
        for index in range(15):
            folds[f"applicable:{index}"] = {"overall": {"ablations": {"fully_predicted": {
                "complete_intervention_groups": 1,
                "complete_intervention_group_accuracy": 0.6,
            }}}}
        for index in range(11):
            folds[f"na:{index}"] = {"overall": {"ablations": {"fully_predicted": {
                "complete_intervention_groups": 0,
                "complete_intervention_group_accuracy": None,
            }}}}
        report = scope_correct_report({
            "primary_transfer_gates": {"checks": checks},
            "primary_folds": folds,
        })
        self.assertTrue(report["passed"])
        self.assertEqual(report["checks"][-1]["value"], 0.6)
        self.assertEqual(len(report["checks"][-1]["not_applicable_folds"]), 11)


if __name__ == "__main__":
    unittest.main()
