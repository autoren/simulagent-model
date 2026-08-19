from __future__ import annotations

import unittest

from v187_clean_typed_clarification_planner import build_problem, evaluate_adaptive, solve_exact


class V187CleanPlannerTest(unittest.TestCase):
    def test_exact_two_bit_tree(self) -> None:
        questions = {"questions": [
            {"question_id": "q0", "family": "bit", "value": 0},
            {"question_id": "q1", "family": "bit", "value": 1},
        ]}
        vectors = {"a": [0, 0], "b": [0, 1], "c": [1, 0], "d": [1, 1]}
        bindings = {"bindings": [
            {"record_id": key, "observation_available": True, "target_contract_id": key}
            for key in vectors
        ]}
        config = {"problem": {
            "maximumTypedQuestionCount": 2, "typedQuestionCost": 0.1,
            "genericTrustedClarificationCost": 0.4, "safeDeferralCost": 0.5,
        }}
        problem = build_problem(questions, vectors, bindings, config)
        solver = solve_exact(problem)
        self.assertEqual(float(solver["value"]), 0.2)
        for target in vectors:
            row = evaluate_adaptive(problem, solver, target)
            self.assertTrue(row["final_exact"])
            self.assertEqual(row["terminal_mode"], "TYPED_SINGLETON")
            self.assertEqual(row["question_count"], 2)

    def test_duplicate_columns_keep_first(self) -> None:
        questions = {"questions": [
            {"question_id": "q0", "family": "x", "value": 0},
            {"question_id": "q0_duplicate", "family": "x", "value": 1},
        ]}
        vectors = {"a": [0, 0], "b": [1, 1]}
        bindings = {"bindings": [
            {"record_id": "a", "observation_available": True, "target_contract_id": "a"},
            {"record_id": "b", "observation_available": True, "target_contract_id": "b"},
        ]}
        config = {"problem": {
            "maximumTypedQuestionCount": 1, "typedQuestionCost": 0.1,
            "genericTrustedClarificationCost": 0.4, "safeDeferralCost": 0.5,
        }}
        problem = build_problem(questions, vectors, bindings, config)
        self.assertEqual([q.question_id for q in problem["questions"]], ["q0"])


if __name__ == "__main__":
    unittest.main()
