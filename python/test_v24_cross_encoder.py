import json
import unittest
from collections import Counter
from pathlib import Path

import numpy as np

from audit_v22r2_grounding import read_jsonl_directory
from audit_v24_cross_encoder import audit
from build_v24_cross_encoder_proposals import build_pairs
from evaluate_v24_cross_encoder import fit_heads, predict_scenes
from v22_relational import canonical_json
from v24_cross_encoder import (
    cross_prompt_layout,
    old_match_scores,
    proposal_candidate_ids,
    sha256_text,
)


class V24CrossEncoderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = json.loads(Path("configs/v24-cross-encoder.json").read_text())
        cls.pairs = build_pairs(cls.config)
        original_lock = json.loads(Path(cls.config["sourceV22r2Lock"]).read_text())
        cls.scenes = read_jsonl_directory(Path(original_lock["source"]["dataset"]) / "scenes")
        cls.predictions = [
            json.loads(line) for line in Path(cls.config["sourcePredictions"]).read_text().splitlines()
            if line.strip()
        ]

    def test_proposal_union_deduplicates_and_preserves_hard_edge(self):
        ids = ["a", "b", "c", "d"]
        rows = proposal_candidate_ids(ids, np.asarray([.4, .3, .2, .1]), "d", 3)
        self.assertEqual([row["candidate_id"] for row in rows], ["a", "b", "c", "d"])
        self.assertEqual(rows[-1]["proposal_sources"], ["global_hard_assignment"])
        rows = proposal_candidate_ids(ids, np.asarray([.4, .3, .2, .1]), "b", 3)
        self.assertEqual(len(rows), 3)
        self.assertIn("global_hard_assignment", rows[1]["proposal_sources"])

    def test_old_match_score_order_is_stable(self):
        candidates = np.asarray([[1., 0.], [0., 1.]], dtype=np.float32)
        evidence = np.asarray([1., 0.], dtype=np.float32)
        scores = old_match_scores(
            candidates, evidence, np.asarray([-1., -1., 1., 1.]), 0.
        )
        self.assertGreater(scores[0], scores[1])

    def test_candidate_fact_is_after_evidence_and_span_is_exact(self):
        pair = self.pairs[0]
        prompt, (start, end) = cross_prompt_layout(pair)
        self.assertEqual(prompt[start:end], pair["agent_input"]["candidate_statement"])
        self.assertLess(prompt.index("Evidence statement:"), prompt.index("Candidate fact:"))

    def test_registered_pair_population_and_proposal_sizes(self):
        self.assertEqual(len(self.pairs), 13372)
        groups = Counter((row["scene_id"], row["evidence_id"]) for row in self.pairs)
        self.assertEqual(Counter(groups.values()), {3: 4376, 4: 61})

    def test_in_memory_audit_passes(self):
        ordered = sorted(self.pairs, key=lambda row: row["id"])
        manifest = {
            "pairs": len(self.pairs),
            "config": "configs/v24-cross-encoder.json",
            "config_sha256": __import__("v10_protocol").file_sha256(
                Path("configs/v24-cross-encoder.json")
            ),
            "corpus_sha256": sha256_text(
                "".join(canonical_json(row) + "\n" for row in ordered)
            ),
            "source_hashes": {},
        }
        result = audit(self.pairs, self.config, manifest, self.scenes, self.predictions)
        self.assertTrue(result["passed"], result["errors"])
        coverage = result["proposal"]["gold_coverage_by_split_and_role"]
        self.assertGreaterEqual(coverage["grounding_evaluation"]["support"], .95)
        self.assertGreaterEqual(coverage["grounding_evaluation"]["query"], .95)
        self.assertEqual(result["proposal"]["perfect_matching_scenes"], 384)

    def test_factorized_heads_and_sparse_assignment_round_trip(self):
        truth_index = {"false": 0, "true": 1, "unknown": 2}
        features = []
        for row in sorted(self.pairs, key=lambda value: value["id"]):
            vector = np.zeros(4, dtype=np.float32)
            vector[0] = 5.0 if row["target"]["same_atom"] else -5.0
            if row["target"]["same_atom"]:
                vector[1 + truth_index[row["target"]["truth_label"]]] = 5.0
            features.append(vector)
        ordered = sorted(self.pairs, key=lambda value: value["id"])
        arrays = {
            "pair_ids": np.asarray([row["id"] for row in ordered]),
            "pair_features": np.stack(features),
        }
        match, truth, diagnostics = fit_heads(self.pairs, arrays, self.config)
        self.assertEqual(diagnostics["truth_multiclass_strategy"], "explicit_one_vs_rest")
        by_scene = {}
        for row in self.pairs:
            by_scene.setdefault(row["scene_id"], []).append(row)
        scene = next(
            scene for scene in self.scenes
            if all(
                any(row["target"]["same_atom"] for row in by_scene[scene["id"]]
                    if row["evidence_id"] == evidence["id"])
                for evidence in scene["agent_input"]["evidence"]
            )
        )
        prediction = predict_scenes([scene], self.pairs, arrays, match, truth)[0]
        target = {
            row["evidence_id"]: (row["candidate_id"], row["truth_label"])
            for row in scene["target"]["atom_groundings"]
        }
        self.assertTrue(all(
            (row["candidate_id"], row["truth_label"]) == target[row["evidence_id"]]
            for row in prediction["rows"]
        ))


if __name__ == "__main__":
    unittest.main()
