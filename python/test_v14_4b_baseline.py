import unittest

from evaluate_v14_4b_baseline import decision, transfer_gate_report


def fold(accuracy, surface=None):
    return {
        "overall": {"accuracy": accuracy},
        "by_surface": {"canonical": {"accuracy": accuracy if surface is None else surface}},
    }


class V14BaselineTests(unittest.TestCase):
    def test_context_is_excluded_from_transfer_gate(self):
        values = {"context": fold(0.0), "surface:a": fold(0.8, 0.7)}
        report = transfer_gate_report(values, {
            "minimumEveryTransferFoldAccuracy": 0.7,
            "minimumEveryTransferSurfaceAccuracy": 0.65,
        })
        self.assertTrue(report["passed"])

    def test_decision_requires_both_worst_case_gates(self):
        self.assertEqual(
            decision({"passed": True}),
            "operator_supported_surface_transfer_passes_repair_temporal_then_full_pipeline",
        )
        self.assertEqual(
            decision({"passed": False}),
            "operator_supported_surface_transfer_fails_audit_before_adaptation",
        )


if __name__ == "__main__":
    unittest.main()
