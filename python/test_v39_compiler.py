import copy
import json
import unittest

from v22r2_grounding import PROJECT_ROOT
from generate_v39_compiler import build_populations
from v39_compiler import compile_agent_input


class V39CompilerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        config = json.loads((PROJECT_ROOT / "configs/v39-declared-language-compiler.json").read_text())
        v32 = json.loads((PROJECT_ROOT / "configs/v32-factorized-semantics.json").read_text())
        cls.populations = build_populations(config, v32)

    def test_population_and_compositional_split(self):
        self.assertEqual(len(self.populations["compiler_development"]), 360)
        self.assertEqual(len(self.populations["supported_evaluation"]), 360)
        self.assertEqual(len(self.populations["compiler_safety"]), 240)
        self.assertEqual(len(self.populations["novel_paraphrase_diagnostic"]), 50)
        dev = {row["oracle_metadata"]["composition_cell"] for row in self.populations["compiler_development"]}
        evaluation = {row["oracle_metadata"]["composition_cell"] for row in self.populations["supported_evaluation"]}
        self.assertEqual(dev, evaluation)
        self.assertEqual(len(evaluation), 120)
        dev_surfaces = {row["agent_input"]["evidence_text"] for row in self.populations["compiler_development"]}
        eval_surfaces = {row["agent_input"]["evidence_text"] for row in self.populations["supported_evaluation"]}
        self.assertFalse(dev_surfaces & eval_surfaces)

    def test_development_exact_compilation(self):
        for row in self.populations["compiler_development"]:
            result = compile_agent_input(row["agent_input"])
            self.assertEqual(result["status"], "ok")
            self.assertEqual(result["parse"], row["target"]["parse"])

    def test_safety_challenges_fail_closed(self):
        row = self.populations["compiler_development"][0]
        metadata = row["oracle_metadata"]
        cases = []
        malformed = copy.deepcopy(row["agent_input"])
        malformed["evidence_text"] = malformed["evidence_text"].replace("Operation cue: ", "Operator clue: ", 1)
        cases.append((malformed, {"abstain"}))
        ambiguous = copy.deepcopy(row["agent_input"])
        ambiguous["evidence_text"] = f"Focal report: {metadata['focus_text']}; Focal report: {metadata['decoy_text']}; Operation cue: {metadata['cue']}; Context only: duplicate."
        cases.append((ambiguous, {"ambiguous", "abstain"}))
        unknown_predicate = copy.deepcopy(row["agent_input"])
        unknown_predicate["evidence_text"] = unknown_predicate["evidence_text"].replace(metadata["focus_text"], "nexa glimmers beside pavo", 1)
        cases.append((unknown_predicate, {"abstain"}))
        unknown_operator = copy.deepcopy(row["agent_input"])
        unknown_operator["evidence_text"] = unknown_operator["evidence_text"].replace(metadata["cue"], "speculation", 1)
        cases.append((unknown_operator, {"abstain"}))
        for agent_input, expected in cases:
            self.assertIn(compile_agent_input(agent_input)["status"], expected)

    def test_compiler_has_no_target_channel(self):
        row = self.populations["compiler_development"][0]
        altered = copy.deepcopy(row)
        altered["target"] = {"parse": {"sentinel": True}}
        self.assertEqual(compile_agent_input(row["agent_input"]), compile_agent_input(altered["agent_input"]))


if __name__ == "__main__":
    unittest.main()
