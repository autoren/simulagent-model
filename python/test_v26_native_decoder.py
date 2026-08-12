import json
import unittest
from collections import Counter
from pathlib import Path

from audit_v22r2_grounding import read_jsonl_directory
from audit_v26_native_decoder import audit
from build_v26_native_decoder import build_rows
from v10_protocol import file_sha256
from v22_relational import canonical_json
from v24_cross_encoder import sha256_text
from v26_native_decoder import decoder_prompt, select_label


class V26NativeDecoderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = json.loads(Path("configs/v26-native-truth-decoder.json").read_text())
        cls.rows = build_rows(cls.config)
        v25_lock = json.loads(Path(cls.config["sourceV25Lock"]).read_text())
        v24_lock = json.loads(Path(v25_lock["source"]["v24_lock"]).read_text())
        original_lock = json.loads(Path(v24_lock["source"]["v22r2_lock"]).read_text())
        cls.scenes = read_jsonl_directory(Path(original_lock["source"]["dataset"]) / "scenes")
        cls.predictions = [
            json.loads(line) for line in Path(cls.config["sourceV24Predictions"]).read_text().splitlines()
            if line.strip()
        ]

    def test_registered_population(self):
        self.assertEqual(len(self.rows), 4437)
        self.assertEqual(Counter(row["split"] for row in self.rows), {
            "grounding_fit": 1482,
            "grounding_calibration": 726,
            "grounding_evaluation": 2229,
        })

    def test_prompt_order_and_firewall(self):
        prompt = decoder_prompt(self.rows[0])
        self.assertLess(prompt.index("Evidence statement:"), prompt.index("Candidate fact:"))
        self.assertTrue(prompt.endswith("Classification:"))
        self.assertNotIn("truth_label", prompt)

    def test_registered_argmax_and_tie_order(self):
        labels = self.config["labels"]
        self.assertEqual(select_label([0.1, 0.9, 0.2], labels)["token"], "B")
        self.assertEqual(select_label([0.5, 0.5, 0.1], labels)["token"], "A")

    def test_in_memory_audit_passes(self):
        ordered = sorted(self.rows, key=lambda row: row["id"])
        manifest = {
            "config": "configs/v26-native-truth-decoder.json",
            "config_sha256": file_sha256(Path("configs/v26-native-truth-decoder.json")),
            "rows": len(self.rows),
            "corpus_sha256": sha256_text("".join(
                canonical_json(row) + "\n" for row in ordered
            )),
            "source_hashes": {},
        }
        result = audit(self.rows, self.config, manifest, self.scenes, self.predictions)
        self.assertTrue(result["passed"], result["errors"])
        self.assertEqual(result["fixed_assignment"]["coverage"], 1.0)


if __name__ == "__main__":
    unittest.main()
