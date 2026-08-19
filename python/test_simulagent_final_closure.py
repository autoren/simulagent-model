#!/usr/bin/env python3
"""Integrity tests for the final Simulagent closure boundary."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "simulagent-final-closure.json"
RESULT = ROOT / "outputs" / "simulagent-final-closure" / "result.json"
PARTICIPANT = (
    ROOT
    / "data"
    / "prospective-language-pilot"
    / "prospective-language-pilot-v1"
    / "P001"
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class FinalClosureTest(unittest.TestCase):
    def setUp(self) -> None:
        self.config = json.loads(CONFIG.read_text(encoding="utf-8"))
        self.result = json.loads(RESULT.read_text(encoding="utf-8"))

    def test_project_is_closed_and_mutating_authorizations_are_false(self) -> None:
        self.assertEqual(self.config["project_status"], "closed")
        allowed = self.config["authorization"]
        self.assertTrue(allowed["documentation_and_non_mutating_reproduction"])
        for name, value in allowed.items():
            if name != "documentation_and_non_mutating_reproduction":
                self.assertFalse(value, name)

    def test_locked_pilot_sources_match_closure(self) -> None:
        sources = {
            "public_requests_sha256": PARTICIPANT / "public" / "phase1_initial_requests.jsonl",
            "phase2_result_sha256": PARTICIPANT / "assistant" / "phase2_architecture" / "result.json",
            "phase3_state_sha256": PARTICIPANT / "private" / "phase3_clarification_state.json",
        }
        expected = {
            "public_requests_sha256": self.config["prospective_pilot"]["phase1"]["public_requests_sha256"],
            "phase2_result_sha256": self.config["prospective_pilot"]["phase2"]["result_sha256"],
            "phase3_state_sha256": self.config["prospective_pilot"]["phase3"]["state_sha256"],
        }
        for name, path in sources.items():
            self.assertEqual(sha256(path), expected[name], name)

    def test_phase3_is_stopped_after_one_locked_unable_response(self) -> None:
        state = json.loads(
            (PARTICIPANT / "private" / "phase3_clarification_state.json").read_text(encoding="utf-8")
        )
        self.assertEqual(len(state["responses"]), 1)
        response = state["responses"]["H003"]
        self.assertEqual(response["response_status"], "unable_to_answer")
        self.assertEqual(response["unable"]["reason"], "do_not_know")
        phase3 = self.config["prospective_pilot"]["phase3"]
        self.assertEqual(phase3["locked_response_count"], 1)
        self.assertEqual(phase3["remaining_response_count"], 10)
        self.assertFalse(phase3["terminal_run_authorized"])

    def test_no_terminal_artifact_exists(self) -> None:
        forbidden_fragments = ("terminal", "phase_4")
        matching = [
            path
            for path in PARTICIPANT.rglob("*")
            if path.is_file() and any(fragment in path.name.lower() for fragment in forbidden_fragments)
        ]
        self.assertEqual(matching, [])

    def test_docs_and_ui_expose_final_closure(self) -> None:
        index = (ROOT / "docs" / "README.md").read_text(encoding="utf-8")
        tracks = (ROOT / "docs" / "simulagent-unpursued-tracks.md").read_text(encoding="utf-8")
        app = (ROOT / "python" / "prospective_language_pilot_app.py").read_text(encoding="utf-8")
        self.assertIn("Simulagent is closed", index)
        self.assertIn("Executable fictional-world clarification oracle", tracks)
        self.assertIn("FINAL_CLOSURE_PATH", app)
        self.assertIn("show_final_closure", app)

    def test_machine_readable_result_matches_config(self) -> None:
        self.assertEqual(self.result["project_status"], "closed")
        self.assertEqual(self.result["decision"], self.config["decision"])
        self.assertEqual(self.result["prospective_pilot"]["phase3"]["locked_response_count"], 1)
        self.assertEqual(self.result["verification"]["status"], "pass")


if __name__ == "__main__":
    unittest.main()

