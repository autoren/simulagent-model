import json
import unittest
from collections import Counter
from pathlib import Path

from audit_v22r2_grounding import read_jsonl_directory
from audit_v24_cross_encoder import read_pairs
from audit_v27_support_map import audit
from build_v27_support_edges import build_rows
from v10_protocol import file_sha256
from v22_relational import canonical_json
from v24_cross_encoder import sha256_text
from v27_support_map import log_softmax, select_episode_map


class V27SupportMapTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = json.loads(Path("configs/v27-support-map.json").read_text())
        cls.rows, cls.support_pairs, _ = build_rows(cls.config)
        v26_lock = json.loads(Path(cls.config["sourceV26Lock"]).read_text())
        v25_lock = json.loads(Path(v26_lock["source"]["v25_lock"]).read_text())
        v24_lock = json.loads(Path(v25_lock["source"]["v24_lock"]).read_text())
        original_lock = json.loads(Path(v24_lock["source"]["v22r2_lock"]).read_text())
        cls.scenes = read_jsonl_directory(Path(original_lock["source"]["dataset"]) / "scenes")
        cls.scores = [
            json.loads(line) for line in Path(cls.config["sourceV26Scores"]).read_text().splitlines()
            if line.strip()
        ]

    def test_registered_edge_partition(self):
        self.assertEqual(len(self.support_pairs), 1652)
        self.assertEqual(len(self.rows), 1103)
        self.assertEqual(Counter(row["split"] for row in self.rows), {
            "grounding_fit": 372, "grounding_calibration": 156,
            "grounding_evaluation": 575,
        })

    def test_in_memory_audit_passes(self):
        ordered = sorted(self.rows, key=lambda row: row["id"])
        manifest = {
            "config": "configs/v27-support-map.json",
            "config_sha256": file_sha256(Path("configs/v27-support-map.json")),
            "new_decoder_rows": len(self.rows),
            "corpus_sha256": sha256_text("".join(
                canonical_json(row) + "\n" for row in ordered
            )),
            "source_hashes": {},
        }
        result = audit(
            self.rows, self.config, manifest, self.scenes, self.support_pairs, self.scores
        )
        self.assertTrue(result["passed"], result["errors"])
        self.assertEqual(result["proposal"]["maximum_sparse_perfect_matchings"], 256)

    def test_log_softmax_and_map_ties_are_deterministic(self):
        values = log_softmax([1.0, 2.0, 3.0])
        self.assertAlmostEqual(sum(__import__("math").exp(value) for value in values), 1.0)
        class Hypothesis:
            def __init__(self, key): self.key = key
        graphs = [[{"log_score": 2.0}, {"log_score": 1.0}]]
        compatibility = [__import__("numpy").asarray([[True, True], [True, True]])]
        selected = select_episode_map(
            [Hypothesis("b"), Hypothesis("a")], graphs, compatibility
        )
        self.assertEqual(selected["program_key"], "a")
        self.assertEqual(selected["graph_indices"], (0,))


if __name__ == "__main__":
    unittest.main()
