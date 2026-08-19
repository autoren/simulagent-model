from __future__ import annotations

from io import BytesIO
import json
import unittest
import zipfile

from v102_presto_context_source import (
    build_presto_context_inventory,
    contains_phrase,
    evaluate_presto_source_gates,
    parse_presto_archive,
    target_arguments,
)


RULE = {
    "minimumNormalizedArgumentCharacterCount": 3,
    "maximumNormalizedArgumentTokenCount": 8,
    "prohibitedNormalizedArguments": ["yes", "no", "true", "false", "none", "null"],
    "eligibleContextSources": [
        "previous_turn_user_query", "previous_turn_response_text", "seeded_list_name",
        "seeded_list_item", "seeded_note_name", "seeded_note_text", "seeded_contact",
    ],
}


CONFIG = {
    "locale": "en-US", "requiredContextProvenance": "human",
    "requiredRecordFields": ["inputs", "targets", "metadata"],
    "requiredMetadataFields": [
        "example_id", "locale", "split", "context", "linguistic_phenomena",
        "previous_turns", "seeded_lists", "seeded_notes", "seeded_contacts",
    ],
    "allowedSourceSplits": ["dev", "test"],
    "canonicalSplitMap": {"dev": "development", "test": "protected_test"},
    "dependencyRule": RULE,
    "sourceGates": {
        "minimumEligibleDevelopmentCandidateCount": 1,
        "minimumEligibleProtectedTestCandidateCount": 1,
        "minimumEligibleTotalCandidateCount": 2,
        "minimumPreviousTurnDependentCandidateCount": 1,
        "minimumSeededStateDependentCandidateCount": 1,
        "minimumDependencySourceKindCount": 2,
        "minimumSemanticRootFunctionCount": 2,
        "maximumSyntheticContextCandidateCount": 0,
    },
}


def row(identifier: str, split: str, target_value: str, *, seeded: bool) -> dict:
    return {
        "inputs": "Use that one", "targets": f"Action_{identifier} ( value « {target_value} » )",
        "metadata": {
            "example_id": identifier, "locale": "en-US", "split": split,
            "context": "human", "linguistic_phenomena": "",
            "previous_turns": [] if seeded else [{
                "user_query": f"Remember {target_value}", "response_text": "Okay",
            }],
            "seeded_lists": [{"name": "Tasks", "items": [target_value]}] if seeded else [],
            "seeded_notes": [], "seeded_contacts": [],
        },
    }


class V102PrestoTests(unittest.TestCase):
    def test_phrase_and_target_parsing_are_normalized_and_contiguous(self) -> None:
        self.assertTrue(contains_phrase("Call José-Luis now", "José Luis"))
        self.assertFalse(contains_phrase("call Jo then Luis", "Jo Luis"))
        self.assertEqual(target_arguments("X ( v « Alpha Beta » )", RULE), ("Alpha Beta",))

    def test_archive_requires_exact_dev_and_test_members(self) -> None:
        buffer = BytesIO()
        with zipfile.ZipFile(buffer, "w") as archive:
            archive.writestr("presto/presto_dev.jsonl", json.dumps(row("d", "dev", "alpha", seeded=False)) + "\n")
            archive.writestr("presto/presto_test.jsonl", json.dumps(row("t", "test", "beta", seeded=True)) + "\n")
        records, members = parse_presto_archive(
            buffer.getvalue(), ["presto_dev.jsonl", "presto_test.jsonl"]
        )
        self.assertEqual(len(records), 2)
        self.assertEqual(set(members), {"presto_dev.jsonl", "presto_test.jsonl"})

    def test_inventory_constructs_paired_text_free_candidates(self) -> None:
        source = [
            ("presto_dev.jsonl", row("d", "dev", "alpha project", seeded=False)),
            ("presto_test.jsonl", row("t", "test", "beta task", seeded=True)),
        ]
        inventory = build_presto_context_inventory(source, CONFIG)
        self.assertEqual(inventory["eligible_candidate_count"], 2)
        self.assertEqual(inventory["previous_turn_dependent_candidate_count"], 1)
        self.assertEqual(inventory["seeded_state_dependent_candidate_count"], 1)
        self.assertTrue(all(evaluate_presto_source_gates(inventory, CONFIG).values()))
        self.assertNotIn("alpha project", str(inventory))
        self.assertNotIn("beta task", str(inventory))
        self.assertFalse(inventory["contains_input_target_argument_context_tokens_seeded_values_or_prompts"])

    def test_argument_present_in_current_input_is_ineligible(self) -> None:
        value = row("d", "dev", "alpha project", seeded=False)
        value["inputs"] = "Use alpha project"
        inventory = build_presto_context_inventory([("presto_dev.jsonl", value)], CONFIG)
        self.assertEqual(inventory["eligible_candidate_count"], 0)


if __name__ == "__main__":
    unittest.main()
