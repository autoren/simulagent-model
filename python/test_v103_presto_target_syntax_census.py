from __future__ import annotations

import copy
import unittest

from test_v102_presto_context_source import CONFIG, row
from v103_presto_target_syntax_census import build_target_syntax_census, evaluate_target_syntax_gates


DIAGNOSTIC = {
    "literalFamilies": [
        "guillemet", "single_guillemet", "ascii_double_quote", "curly_double_quote",
        "ascii_single_quote", "square_bracket",
    ],
    "candidateEligibleLiteralFamilies": [
        "guillemet", "single_guillemet", "ascii_double_quote", "curly_double_quote",
        "ascii_single_quote",
    ],
    "diagnosticStages": [
        "record_contains_literal_family", "quality_filtered_literal",
        "literal_absent_from_current_input", "literal_present_in_context",
        "literal_absent_from_input_and_present_in_context",
    ],
    "structuralCharacterFeatures": [
        "left_parenthesis", "right_parenthesis", "colon", "equals", "left_square_bracket",
        "left_guillemet", "left_single_guillemet", "ascii_double_quote", "curly_double_quote",
    ],
    "diagnosticGates": {
        "minimumUnionDevelopmentCandidateCount": 1,
        "minimumUnionProtectedTestCandidateCount": 1,
        "minimumUnionTotalCandidateCount": 2,
        "minimumUnionPreviousTurnDependentCandidateCount": 1,
        "minimumUnionSeededStateDependentCandidateCount": 1,
        "minimumUnionDependencySourceKindCount": 2,
        "minimumUnionSemanticRootFunctionCount": 2,
        "maximumEmittedCandidateIdentifierCount": 0,
    },
}


class V103SyntaxCensusTests(unittest.TestCase):
    def test_multiple_literal_families_are_counted_without_emission(self) -> None:
        dev = row("d", "dev", "alpha project", seeded=False)
        test = row("t", "test", "beta task", seeded=True)
        test["targets"] = 'Action_t ( value "beta task" )'
        census = build_target_syntax_census([
            ("presto_dev.jsonl", dev), ("presto_test.jsonl", test),
        ], CONFIG, DIAGNOSTIC)
        self.assertEqual(census["candidate_eligible_family_union_count"], 2)
        self.assertTrue(all(evaluate_target_syntax_gates(census, DIAGNOSTIC).values()))
        self.assertNotIn("alpha project", str(census))
        self.assertNotIn("beta task", str(census))
        self.assertEqual(census["emitted_candidate_identifier_count"], 0)

    def test_literal_in_current_input_does_not_enter_union(self) -> None:
        value = row("d", "dev", "alpha project", seeded=False)
        value["inputs"] = "Use alpha project"
        census = build_target_syntax_census([
            ("presto_dev.jsonl", value),
        ], CONFIG, DIAGNOSTIC)
        self.assertEqual(census["candidate_eligible_family_union_count"], 0)

    def test_square_brackets_are_diagnostic_not_candidate_eligible(self) -> None:
        value = row("d", "dev", "alpha project", seeded=False)
        value["targets"] = "Action_d [ alpha project ]"
        census = build_target_syntax_census([
            ("presto_dev.jsonl", value),
        ], CONFIG, DIAGNOSTIC)
        self.assertEqual(census["literal_family_stage_record_counts"]["square_bracket"][
            "literal_absent_from_input_and_present_in_context"
        ], 1)
        self.assertEqual(census["candidate_eligible_family_union_count"], 0)


if __name__ == "__main__":
    unittest.main()
