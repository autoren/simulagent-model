import unittest

from v18_schema import (
    allowed_trace_consistent_hypotheses,
    behaviorally_equivalent,
    binary,
    build_program,
    enumerate_program_hypotheses,
    execute_query,
    greedy_distinguishing_support,
    program_signature,
    trace_consistent_hypotheses,
    variable,
    version_space_query,
)


class V18SchemaTests(unittest.TestCase):
    def test_behavioral_equivalence_ignores_commutative_surface_order(self) -> None:
        left = {
            "dsl_version": 1,
            "determinants": [
                {"id": "a", "type": "boolean"},
                {"id": "b", "type": "boolean"},
            ],
            "output_bits": [binary("and", variable("a"), variable("b"))],
        }
        right = {
            **left,
            "output_bits": [
                {"op": "and", "args": [variable("b"), variable("a")]}
            ],
        }
        self.assertTrue(behaviorally_equivalent(left, right))

    def test_query_distinguishes_sensitive_and_invariant_unknowns(self) -> None:
        program = build_program(("a", "b"), (("var", ("a",)),))
        sensitive = execute_query(program, [
            {"determinant_id": "a", "allowed_values": ["inactive", "active"]},
            {"determinant_id": "b", "allowed_values": ["inactive"]},
        ])
        invariant = execute_query(program, [
            {"determinant_id": "a", "allowed_values": ["active"]},
            {"determinant_id": "b", "allowed_values": ["inactive", "active"]},
        ])
        self.assertFalse(sensitive["identifiable"])
        self.assertEqual(sensitive["possible_transition_codes"], ["transition_0", "transition_1"])
        self.assertTrue(invariant["identifiable"])
        self.assertEqual(invariant["possible_transition_codes"], ["transition_1"])

    def test_greedy_support_recovers_one_behavioral_class(self) -> None:
        determinant_ids = ("a", "b", "c")
        target = build_program(
            determinant_ids,
            (("or_of_and", ("a", "b", "c")), ("xor", ("a", "c"))),
        )
        hypotheses = enumerate_program_hypotheses(determinant_ids, 2)
        support = greedy_distinguishing_support(target, hypotheses)
        consistent = trace_consistent_hypotheses(hypotheses, support, determinant_ids)
        self.assertEqual(len(consistent), 1)
        self.assertEqual(consistent[0].signature, program_signature(target))
        self.assertLessEqual(len(support), 8)

    def test_version_space_preserves_program_uncertainty(self) -> None:
        determinant_ids = ("a", "b")
        hypotheses = enumerate_program_hypotheses(determinant_ids, 1)
        answer = version_space_query(hypotheses, [
            {"determinant_id": "a", "allowed_values": ["inactive"]},
            {"determinant_id": "b", "allowed_values": ["inactive"]},
        ])
        self.assertGreater(answer["behavioral_hypotheses"], 1)
        self.assertFalse(answer["identifiable"])
        self.assertEqual(answer["possible_transition_codes"], ["transition_0", "transition_1"])

    def test_uncertain_support_uses_existential_assignment_consistency(self) -> None:
        determinant_ids = ("a", "b")
        hypotheses = enumerate_program_hypotheses(determinant_ids, 1)
        target = build_program(determinant_ids, (("var", ("a",)),))
        retained = allowed_trace_consistent_hypotheses(hypotheses, [{
            "allowed_values": [
                {"determinant_id": "a", "allowed_values": ["inactive", "active"]},
                {"determinant_id": "b", "allowed_values": ["inactive"]},
            ],
            "transition_code": "transition_1",
        }], determinant_ids)
        self.assertIn(program_signature(target), {value.signature for value in retained})


if __name__ == "__main__":
    unittest.main()
