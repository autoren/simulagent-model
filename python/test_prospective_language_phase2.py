import json
import unittest
from pathlib import Path

from prospective_language_phase2 import (
    controller_output,
    parse_semantic_proposal,
    render_phase2_user_payload,
    validate_phase2_config,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = json.loads((ROOT / "configs/prospective-language-pilot-v1-phase2.json").read_text())


def raw(**overrides):
    value = {
        "semantic_state": "ONE_CLEAR_READING",
        "candidate_goals": ["Receive a bounded plan"],
        "missing_evidence": [],
        "evidence_source": "NONE_REQUIRED",
        "clarification_questions": [],
        "sandbox_plan": "Here is a useful sandbox-only plan with enough detail.",
        "defer_reason": None,
    }
    value.update(overrides)
    return json.dumps(value)


class ProspectiveLanguagePhase2Tests(unittest.TestCase):
    def test_config_is_frozen_and_nonexecuting(self):
        validate_phase2_config(CONFIG)
        self.assertFalse(CONFIG["controller"]["real_world_execution_enabled"])
        self.assertFalse(CONFIG["post_run_rule"]["authorizes_api_fallback"])

    def test_plan_requires_one_clear_reading_and_no_missing_evidence(self):
        parsed = parse_semantic_proposal(raw(), CONFIG)
        self.assertTrue(parsed["structurally_valid"])
        self.assertEqual("PLAN", controller_output(parsed, CONFIG)["route"])

    def test_user_supplied_evidence_routes_to_one_clarification_wave(self):
        parsed = parse_semantic_proposal(
            raw(
                semantic_state="INSUFFICIENT_OBSERVABLE_EVIDENCE",
                missing_evidence=["LOCAL_FACT"],
                evidence_source="USER_CAN_SUPPLY",
                clarification_questions=["What location should the plan use?"],
                sandbox_plan=None,
            ),
            CONFIG,
        )
        self.assertTrue(parsed["structurally_valid"])
        output = controller_output(parsed, CONFIG)
        self.assertEqual("CLARIFY", output["route"])
        self.assertEqual(1, len(output["clarification_questions"]))

    def test_unavailable_external_evidence_routes_to_defer(self):
        parsed = parse_semantic_proposal(
            raw(
                semantic_state="INSUFFICIENT_OBSERVABLE_EVIDENCE",
                missing_evidence=["SOURCE_CONTENT"],
                evidence_source="EXTERNAL_SOURCE_REQUIRED",
                sandbox_plan=None,
                defer_reason="The source contents were not supplied and cannot be inspected offline.",
            ),
            CONFIG,
        )
        self.assertTrue(parsed["structurally_valid"])
        self.assertEqual("DEFER", controller_output(parsed, CONFIG)["route"])

    def test_invalid_json_uses_safe_defer_without_retry(self):
        parsed = parse_semantic_proposal("not-json", CONFIG)
        self.assertFalse(parsed["structurally_valid"])
        output = controller_output(parsed, CONFIG)
        self.assertEqual("DEFER", output["route"])
        self.assertTrue(output["used_safe_fallback"])

    def test_multiple_question_marks_are_rejected(self):
        parsed = parse_semantic_proposal(
            raw(
                semantic_state="MULTIPLE_PLAUSIBLE_READINGS",
                missing_evidence=["USER_GOAL"],
                evidence_source="USER_CAN_SUPPLY",
                clarification_questions=["Which outcome matters? And why?"],
                sandbox_plan=None,
            ),
            CONFIG,
        )
        self.assertFalse(parsed["structurally_valid"])

    def test_prompt_contains_only_registered_visible_fields(self):
        record = {
            "record_id": "H001",
            "initial_request": "Please help with this situation.",
            "assistant_visible_context": {"domain": "example", "available_support": ["defer"]},
            "private_goal": "must never appear",
        }
        payload = render_phase2_user_payload(record, CONFIG)
        self.assertIn("Please help", payload)
        self.assertNotIn("must never appear", payload)


if __name__ == "__main__":
    unittest.main()
