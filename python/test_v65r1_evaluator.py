#!/usr/bin/env python3
from __future__ import annotations

import copy
import json
import unittest

from v22r2_grounding import PROJECT_ROOT
from evaluate_v65r1_eig import WORK_FIELDS, aggregate_evaluation, weighted_wasserstein_1


CONTROL_NAMES = (
    "average_repeat_EIG",
    "first_repeat_only",
    "state_as_target",
    "map_identity",
    "theta_mean",
    "equal_identity_evidence",
    "plugin_particle_state_predictive",
)


def frozen_config() -> dict:
    design = json.loads((PROJECT_ROOT / "configs/v65r1-design-lock.json").read_text())
    return copy.deepcopy(design["config_payload"])


def synthetic_implementation_audit() -> dict:
    return {
        "mutation_audit": {
            "kill_rate": 1.0,
            "checks": {"share_inner_streams_across_outer_particles": True},
        },
        "analytic_fixtures": {"pass_rate": 1.0},
    }


def synthetic_access() -> dict[str, int]:
    return {
        "logical_evaluation_attempts": 1,
        "subset_public_records_loaded": 48,
        "v64_source_public_records_loaded_during_evaluation": 0,
        "v64_selection_audit_records_loaded": 0,
        "v64_evaluation_records_loaded": 0,
        "truth_field_access_count": 0,
        "realized_outcome_access_before_selection_count": 0,
        "candidate_omission_count": 0,
        "tie_break_violation_count": 0,
        "random_stream_collision_count": 0,
        "human_record_access_count": 0,
        "model_forward_pass_count": 0,
        "adapter_training_run_count": 0,
    }


def synthetic_rows() -> list[dict]:
    rows = []
    errors = {31: 0.0020, 127: 0.0010, 509: 0.0004}
    posterior = {31: 0.035, 127: 0.020, 509: 0.010}
    for record_index in range(48):
        for budget in (31, 127, 509):
            error = errors[budget]
            metric = posterior[budget]
            controls = {
                name: {
                    "values": [0.0, 0.0, 0.0, 0.0],
                    "selected_action": "e",
                    "strict_optimal_membership": False,
                    "epsilon_optimal_membership": False,
                    "exact_regret": 0.002,
                    "exact_optimal_actions": ["n"],
                    "mean_absolute_eig_error": 0.006,
                }
                for name in CONTROL_NAMES
            }
            repeats = []
            for repeat in range(3):
                repeats.append(
                    {
                        "repeat": repeat,
                        "values": [0.020 + error] * 4,
                        "mean_absolute_eig_error": error * 1.2,
                        "selected_action": "n",
                        "strict_optimal_membership": True,
                        "epsilon_optimal_membership": True,
                        "exact_regret": 0.0,
                        "exact_optimal_actions": ["n"],
                        "runtime_seconds": 0.01,
                        "work": {field: 1 for field in WORK_FIELDS},
                        "random_stream_count": 9,
                        "random_stream_collision_count": 0,
                        "normalizes": True,
                    }
                )
            rows.append(
                {
                    "record_id": f"synthetic-{record_index:02d}",
                    "prefix_length": record_index // 8,
                    "budget": budget,
                    "pooled_normalizes": True,
                    "candidate_predictive_normalizes": True,
                    "candidate_count": 4,
                    "candidate_order": ["n", "e", "s", "w"],
                    "tie_break_valid": True,
                    "finite": True,
                    "identity_tv": metric,
                    "theta_wasserstein": metric / 2,
                    "joint_identity_theta_tv": metric * 1.5,
                    "state_tv": metric,
                    "candidate_predictive_tvs": [metric / 2] * 4,
                    "mean_candidate_predictive_tv": metric / 2,
                    "approximate_values": [0.020 + error] * 4,
                    "exact_values": [0.020] * 4,
                    "absolute_eig_errors": [error] * 4,
                    "mean_absolute_eig_error": error,
                    "selected_action": "n",
                    "strict_optimal_membership": True,
                    "epsilon_optimal_membership": True,
                    "exact_regret": 0.0,
                    "exact_optimal_actions": ["n"],
                    "repeat_diagnostics": repeats,
                    "repeat_selected_action_disagreement": False,
                    "best_minus_worst_repeat_regret_spread": 0.0,
                    "controls": controls,
                }
            )
    return rows


class V65r1EvaluatorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = frozen_config()
        self.rows = synthetic_rows()
        self.audit = synthetic_implementation_audit()
        self.access = synthetic_access()

    def evaluate(self, rows=None, audit=None, access=None) -> dict:
        return aggregate_evaluation(
            self.rows if rows is None else rows,
            self.config,
            self.audit if audit is None else audit,
            self.access if access is None else access,
        )

    def test_synthetic_passing_fixture(self) -> None:
        result = self.evaluate()
        self.assertTrue(result["passed"], result["failed_gates"])
        self.assertEqual(result["controls"]["detected_or_dominated"], 9)
        self.assertEqual(len(result["compute_diagnostics"]["cells"]), 432)

    def test_weighted_wasserstein_reference_cases(self) -> None:
        self.assertAlmostEqual(weighted_wasserstein_1([0], [1], [1], [1]), 1.0)
        self.assertAlmostEqual(
            weighted_wasserstein_1([0, 1], [0.5, 0.5], [0, 1], [0.5, 0.5]),
            0.0,
        )

    def test_incomplete_grid_is_rejected(self) -> None:
        self.assertFalse(self.evaluate(rows=self.rows[:-1])["passed"])

    def test_duplicate_record_budget_is_rejected(self) -> None:
        mutant = copy.deepcopy(self.rows)
        mutant[-1] = copy.deepcopy(mutant[0])
        self.assertFalse(self.evaluate(rows=mutant)["passed"])

    def test_missing_repeat_is_rejected(self) -> None:
        mutant = copy.deepcopy(self.rows)
        mutant[0]["repeat_diagnostics"] = mutant[0]["repeat_diagnostics"][:-1]
        self.assertFalse(self.evaluate(rows=mutant)["passed"])

    def test_normalization_and_finite_mutants_are_rejected(self) -> None:
        for field in ("pooled_normalizes", "candidate_predictive_normalizes", "finite"):
            mutant = copy.deepcopy(self.rows)
            mutant[0][field] = False
            self.assertFalse(self.evaluate(rows=mutant)["passed"], field)

    def test_primary_accuracy_mutants_are_rejected(self) -> None:
        for field, value in (
            ("identity_tv", 0.20),
            ("theta_wasserstein", 0.20),
            ("joint_identity_theta_tv", 0.40),
            ("state_tv", 0.30),
        ):
            mutant = copy.deepcopy(self.rows)
            for row in mutant:
                if row["budget"] == 509:
                    row[field] = value
            self.assertFalse(self.evaluate(rows=mutant)["passed"], field)

    def test_primary_predictive_and_eig_mutants_are_rejected(self) -> None:
        predictive = copy.deepcopy(self.rows)
        eig = copy.deepcopy(self.rows)
        for row in predictive:
            if row["budget"] == 509:
                row["candidate_predictive_tvs"] = [0.25] * 4
        for row in eig:
            if row["budget"] == 509:
                row["absolute_eig_errors"] = [0.03] * 4
        self.assertFalse(self.evaluate(rows=predictive)["passed"])
        self.assertFalse(self.evaluate(rows=eig)["passed"])

    def test_selection_mutants_are_rejected(self) -> None:
        for field, value in (
            ("strict_optimal_membership", False),
            ("epsilon_optimal_membership", False),
            ("exact_regret", 0.03),
        ):
            mutant = copy.deepcopy(self.rows)
            for row in mutant:
                if row["budget"] == 509:
                    row[field] = value
            self.assertFalse(self.evaluate(rows=mutant)["passed"], field)

    def test_scaling_mutant_is_rejected(self) -> None:
        mutant = copy.deepcopy(self.rows)
        for row in mutant:
            row["absolute_eig_errors"] = [0.0001 if row["budget"] == 31 else 0.002] * 4
        self.assertFalse(self.evaluate(rows=mutant)["passed"])

    def test_control_mutant_is_rejected(self) -> None:
        mutant = copy.deepcopy(self.rows)
        for row in mutant:
            for control in row["controls"].values():
                control["exact_regret"] = 0.0
                control["strict_optimal_membership"] = True
                control["mean_absolute_eig_error"] = row["mean_absolute_eig_error"]
        self.assertFalse(self.evaluate(rows=mutant)["passed"])

    def test_access_mutants_are_rejected(self) -> None:
        for field, value in (
            ("logical_evaluation_attempts", 2),
            ("subset_public_records_loaded", 47),
            ("v64_source_public_records_loaded_during_evaluation", 1),
            ("v64_selection_audit_records_loaded", 1),
            ("v64_evaluation_records_loaded", 1),
            ("truth_field_access_count", 1),
            ("realized_outcome_access_before_selection_count", 1),
            ("candidate_omission_count", 1),
            ("tie_break_violation_count", 1),
            ("random_stream_collision_count", 1),
            ("human_record_access_count", 1),
            ("model_forward_pass_count", 1),
            ("adapter_training_run_count", 1),
        ):
            access = copy.deepcopy(self.access)
            access[field] = value
            self.assertFalse(self.evaluate(access=access)["passed"], field)

    def test_frozen_implementation_audit_mutants_are_rejected(self) -> None:
        for section, field in (
            ("mutation_audit", "kill_rate"),
            ("analytic_fixtures", "pass_rate"),
        ):
            audit = copy.deepcopy(self.audit)
            audit[section][field] = 0.99
            self.assertFalse(self.evaluate(audit=audit)["passed"])


if __name__ == "__main__":
    unittest.main()
