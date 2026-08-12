import unittest

from audit_v7_shortcuts import gate_report


class V7ShortcutGateTests(unittest.TestCase):
    def test_all_shortcut_ceilings_must_pass(self):
        config = {
            "shortcutGates": {
                "maximumMetadataBalancedAccuracy": 0.55,
                "maximumEvidenceTextBalancedAccuracy": 0.60,
                "maximumEvidenceTextAuc": 0.65,
            }
        }
        metric = {"calibration": {"balanced_accuracy": 0.5, "roc_auc": 0.5}}
        report = gate_report(config, metric, metric)
        self.assertTrue(report["passed"])
        self.assertTrue(all(check["passed"] for check in report["checks"]))

        failed = {"calibration": {"balanced_accuracy": 0.7, "roc_auc": 0.7}}
        report = gate_report(config, failed, failed)
        self.assertFalse(report["passed"])


if __name__ == "__main__":
    unittest.main()
