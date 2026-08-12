import unittest

from audit_v8_shortcuts import shortcut_gate_report


class V8ShortcutGateTests(unittest.TestCase):
    def test_every_shortcut_ceiling_is_hard(self):
        audits = {
            name: {
                "maximum_fold_balanced_accuracy": value,
                "maximum_fold_auc_separation": 0.5,
            }
            for name, value in {
                "metadata": 0.5,
                "unigram": 0.5,
                "character_ngram": 0.61,
                "length": 0.5,
            }.items()
        }
        ceilings = {
            "maximumMetadataWorstFoldBalancedAccuracy": 0.55,
            "maximumUnigramWorstFoldBalancedAccuracy": 0.55,
            "maximumCharacterNgramWorstFoldBalancedAccuracy": 0.6,
            "maximumLengthWorstFoldBalancedAccuracy": 0.55,
            "maximumUnigramWorstFoldAuc": 0.65,
            "maximumCharacterNgramWorstFoldAuc": 0.65,
            "maximumLengthWorstFoldAuc": 0.55,
        }
        report = shortcut_gate_report(audits, ceilings)
        self.assertFalse(report["passed"])
        self.assertEqual(
            [check["name"] for check in report["checks"] if not check["passed"]],
            ["character_ngram_maximum_fold_balanced_accuracy"],
        )


if __name__ == "__main__":
    unittest.main()
