import unittest

from v120_selective_query_value_audit import run_audit


class V120SelectiveQueryValueTests(unittest.TestCase):
    def test_simple_decomposition(self):
        metrics = {"mean_regret_including_clarification": 0.8}
        parent = {"summary": {"baseline_mean_regret": 0.75, "conditions": {"p": {"0.95": {"0.00": {"correlation_aware": metrics}}}}, "outcome_gates": {key: True for key in ("aware_known_exact_every_prior_and_required_correlation", "aware_unsupported_every_prior_and_required_correlation", "aware_false_known_every_prior_and_required_correlation", "misspecified_regret_at_050_every_prior", "misspecified_false_known_at_050_every_prior", "perfect_channel_mean_regret_every_prior", "rho_one_stress_reported", "true_hypothesis_retention", "zero_actual_execution")}}}
        config = {"decomposition": {"fixedClarificationCost": 0.30}, "requiredSlice": {"reliability": "0.95", "planner": "correlation_aware", "correlations": ["0.00"], "expectedFailingConditionIds": ["p@0.00"]}, "outcomeGates": {"requiredFailingConditionCount": 1, "maximumRegretExcessOverBaseline": 0.1, "maximumMinimumZeroLossSkipFraction": 0.2, "maximumIndividualRecordReadCount": 0, "maximumIndividualRecordEmissionCount": 0, "maximumActualExecutionCount": 0}, "decisionRule": {"ifAllOutcomeAndAccessGatesPass": "pass", "otherwise": "fail"}}
        result = run_audit(parent, config)
        self.assertAlmostEqual(result["failing_conditions"][0]["decision_regret_excluding_query_cost"], 0.5)
        self.assertAlmostEqual(result["failing_conditions"][0]["minimum_zero_loss_skip_fraction"], 1 / 6)


if __name__ == "__main__": unittest.main()
