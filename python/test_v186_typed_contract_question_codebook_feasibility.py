from __future__ import annotations

import json
import unittest

from v22r2_grounding import PROJECT_ROOT
from v186_typed_contract_question_codebook_feasibility import answer


class V186TypedContractCodebookTest(unittest.TestCase):
    def test_typed_answers_use_only_semantic_payload(self) -> None:
        contract = {
            "semantic_payload": {
                "normalized_intent_name": "findweather",
                "domain": "weather",
                "slots": [{"name": "city"}],
                "required_slots": ["city"],
                "result_slots": ["temperature"],
                "is_transactional": False,
            }
        }
        self.assertEqual(answer(contract, {"family": "intent_concept", "value": "findweather"}), 1)
        self.assertEqual(answer(contract, {"family": "slot_any", "value": "city"}), 1)
        self.assertEqual(answer(contract, {"family": "slot_result", "value": "city"}), 0)
        self.assertEqual(answer(contract, {"family": "transactional", "value": True}), 0)

    def test_successor_authority_is_closed(self) -> None:
        config = json.loads(
            (PROJECT_ROOT / "configs/v186-typed-contract-question-codebook-feasibility.json").read_text()
        )
        self.assertTrue(config["questionFamilies"]["questionDerivationMayNotUseServiceVersionSourceDefinitionTruthKindPresentedCandidateLanguageOrOutcomes"])
        self.assertFalse(config["decisionRule"]["passAuthorizesImmediatePlannerScoring"])
        self.assertFalse(config["decisionRule"]["passAuthorizesLanguageModelOrProtectedLanguageAccess"])


if __name__ == "__main__":
    unittest.main()
