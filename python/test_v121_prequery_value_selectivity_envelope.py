import unittest
from v121_prequery_value_selectivity_envelope import run_audit


class V121EnvelopeTests(unittest.TestCase):
    def test_envelope_identity(self):
        parent = {"summary": {"fixed_clarification_cost": 0.3, "failing_conditions": [{"condition_id": "x", "decision_value_relative_to_baseline": 0.29, "regret_excess_over_baseline": 0.01}]}}
        config = {"skipFractions": [0.05, 0.10, 0.25, 0.50], "outcomeGates": {"requiredFailingConditionCount": 1, "maximumMinimumQueriedValueLift": 1.1, "maximumFivePercentSkipValueForHardConditions": 1.0, "minimumNonnegativeSkipFractionCountEveryCondition": 4, "requireAggregateMetricsInsufficientToCertifyAnyTrigger": True, "maximumIndividualRecordReadCount": 0, "maximumIndividualRecordEmissionCount": 0, "maximumActualExecutionCount": 0}, "decisionRule": {"ifAllOutcomeAndAccessGatesPass": "pass", "otherwise": "fail"}}
        # The hard-condition set is intentionally empty in this unit fixture; use a real ID to exercise it.
        config["outcomeGates"]["maximumFivePercentSkipValueForHardConditions"] = 0.11
        parent["summary"]["failing_conditions"][0]["condition_id"] = "strong_candidate@0.50"
        result = run_audit(parent, config)
        bound = result["conditions"][0]["skip_value_bounds"][0]
        self.assertAlmostEqual(bound["maximum_average_query_value_in_skipped_subset"], 0.1)
        self.assertAlmostEqual(result["conditions"][0]["minimum_queried_value_lift_ratio"], 0.3 / 0.29)
        self.assertFalse(result["trigger_certified"])


if __name__ == "__main__": unittest.main()
