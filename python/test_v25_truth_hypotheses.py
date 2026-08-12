import json
import unittest
from collections import Counter
from pathlib import Path

from audit_v22r2_grounding import read_jsonl_directory
from audit_v25_truth_hypotheses import audit
from build_v25_truth_hypotheses import build_rows
from evaluate_v25_truth_hypotheses import fit_head, predict_scenes
import numpy as np
from v10_protocol import file_sha256
from v22_relational import canonical_json
from v24_cross_encoder import sha256_text
from v25_truth_hypotheses import truth_prompt_layout


class V25TruthHypothesisTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = json.loads(Path("configs/v25-truth-hypotheses.json").read_text())
        cls.rows = build_rows(cls.config)
        v24_lock = json.loads(Path(cls.config["sourceV24Lock"]).read_text())
        original_lock = json.loads(Path(v24_lock["source"]["v22r2_lock"]).read_text())
        cls.scenes = read_jsonl_directory(Path(original_lock["source"]["dataset"]) / "scenes")
        cls.predictions = [
            json.loads(line) for line in Path(cls.config["sourceV24Predictions"]).read_text().splitlines()
            if line.strip()
        ]

    def test_registered_population(self):
        self.assertEqual(len(self.rows), 13383)
        self.assertEqual(Counter(row["split"] for row in self.rows), {
            "grounding_fit": 4518,
            "grounding_calibration": 2178,
            "grounding_evaluation": 6687,
        })
        self.assertEqual(sum(row["target"]["use_for_fit"] for row in self.rows), 4446)
        self.assertEqual(sum(
            row["target"]["use_for_fit"] and row["target"]["compatible"]
            for row in self.rows
        ), 1482)

    def test_assessment_span_is_final_and_exact(self):
        for row in self.rows[:100]:
            prompt, (start, end) = truth_prompt_layout(row)
            self.assertEqual(prompt[start:end], row["agent_input"]["assessment_statement"])
            self.assertEqual(end, len(prompt))

    def test_each_base_pair_has_all_three_assessments(self):
        groups = {}
        for row in self.rows:
            groups.setdefault((row["scene_id"], row["evidence_id"], row["candidate_id"]), []).append(row)
        expected = {"entailed", "contradicted", "unresolved"}
        self.assertTrue(all({row["assessment_id"] for row in group} == expected for group in groups.values()))
        self.assertTrue(all(sum(row["target"]["compatible"] for row in group) == 1 for group in groups.values()))

    def test_in_memory_audit_passes(self):
        ordered = sorted(self.rows, key=lambda row: row["id"])
        manifest = {
            "config": "configs/v25-truth-hypotheses.json",
            "config_sha256": file_sha256(Path("configs/v25-truth-hypotheses.json")),
            "rows": len(self.rows),
            "corpus_sha256": sha256_text("".join(
                canonical_json(row) + "\n" for row in ordered
            )),
            "source_hashes": {},
        }
        result = audit(self.rows, self.config, manifest, self.scenes, self.predictions)
        self.assertTrue(result["passed"], result["errors"])
        self.assertEqual(result["fixed_assignment"]["coverage"], 1.0)
        self.assertEqual(result["population"]["fit_gold_groups"], 1482)

    def test_binary_compatibility_and_fixed_assignment_round_trip(self):
        ordered = sorted(self.rows, key=lambda row: row["id"])
        arrays = {
            "row_ids": np.asarray([row["id"] for row in ordered]),
            "truth_features": np.asarray([
                [5.0 if row["target"]["compatible"] else -5.0] for row in ordered
            ], dtype=np.float32),
        }
        head, diagnostics = fit_head(self.rows, arrays, self.config)
        self.assertEqual(diagnostics["rows"], 4446)
        scene = self.scenes[0]
        prediction = predict_scenes([scene], self.rows, arrays, head, self.config)[0]
        source = next(row for row in self.predictions if row["scene_id"] == scene["id"])
        source_candidate = {row["evidence_id"]: row["candidate_id"] for row in source["rows"]}
        target_truth = {
            row["evidence_id"]: row["truth_label"] for row in scene["target"]["atom_groundings"]
        }
        self.assertTrue(all(
            row["candidate_id"] == source_candidate[row["evidence_id"]]
            and row["truth_label"] == target_truth[row["evidence_id"]]
            for row in prediction["rows"]
        ))


if __name__ == "__main__":
    unittest.main()
