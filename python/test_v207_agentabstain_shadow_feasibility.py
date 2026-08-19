from __future__ import annotations

import json
import unittest
from pathlib import Path

from v207_agentabstain_shadow_feasibility import (
    derive_tree_pairs,
    extracted_identifiers,
    selected_code_schema_paths,
)


CONFIG = json.loads(Path("configs/v207-agentabstain-shadow-feasibility.json").read_text())


class V207SchemaRulesTest(unittest.TestCase):
    def test_schema_path_allowlist_excludes_unapproved_files(self) -> None:
        paths = [
            "src/configs/tasks.yaml",
            "src/types/task.py",
            "eval/evaluators/commit.py",
            "data/task_payload.json",
            "README.md",
        ]
        self.assertEqual(
            selected_code_schema_paths(paths, CONFIG),
            ["eval/evaluators/commit.py", "src/configs/tasks.yaml", "src/types/task.py"],
        )

    def test_tree_pair_derivation_requires_both_sides(self) -> None:
        paths = [
            "tasks/pre_execution_missing_input/pair01_should_act.json",
            "tasks/pre_execution_missing_input/pair01_should_abstain.json",
            "tasks/pre_execution_conflict/pair02_should_act.json",
            "tasks/runtime_failure/pair03_should_abstain.json",
        ]
        result = derive_tree_pairs(paths, CONFIG["fixedMetadataPatterns"])
        self.assertEqual(result["complete_pair_count"], 1)
        self.assertEqual(result["preexecution_pair_count"], 1)
        self.assertEqual(result["preexecution_scenario_count"], 1)

    def test_identifier_extraction_returns_names_not_values(self) -> None:
        identifiers = extracted_identifiers(b"class Task: task_id: str; should_abstain: bool; prompt: str")
        self.assertTrue({"task_id", "should_abstain", "prompt"}.issubset(identifiers))


if __name__ == "__main__":
    unittest.main()
