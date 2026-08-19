from __future__ import annotations

import json
import unittest

from v22r2_grounding import PROJECT_ROOT
from v174_certificate_depth_feasibility_census import evaluate_feasibility


class V174CertificateDepthFeasibilityCensusTest(unittest.TestCase):
    def test_one_state_certificate_census_is_exact_and_monotone(self) -> None:
        states_all = json.loads(
            (PROJECT_ROOT / "outputs/v172-trusted-shadow-integration-population/population/constraint-states.json").read_text()
        )["states"]
        eligible_ids = json.loads(
            (PROJECT_ROOT / "outputs/v172-trusted-shadow-integration-population/population/integration-eligible-state-ids.json").read_text()
        )["state_ids"]
        targets_all = json.loads(
            (PROJECT_ROOT / "outputs/v172-trusted-shadow-integration-population/population/target-cases.json").read_text()
        )["target_cases"]
        state_id = eligible_ids[0]
        states = {"states": [next(row for row in states_all if row["state_id"] == state_id)]}
        eligible = {"state_ids": [state_id]}
        targets = {"target_cases": [row for row in targets_all if row["state_id"] == state_id]}
        planner_config = json.loads(
            (PROJECT_ROOT / "configs/v167-exact-evidence-gathering-planner-lock.json").read_text()
        )["config_payload"]
        evaluation = evaluate_feasibility(states, eligible, targets, planner_config, list(range(6)))
        self.assertEqual(evaluation["summary"]["state_count"], 1)
        self.assertEqual(evaluation["summary"]["target_count"], 32)
        self.assertEqual(evaluation["summary"]["certificate_validity_rate"], 1.0)
        self.assertEqual(evaluation["summary"]["certificate_minimality_rate"], 1.0)
        self.assertEqual(evaluation["summary"]["horizon_monotonicity_rate"], 1.0)


if __name__ == "__main__":
    unittest.main()
