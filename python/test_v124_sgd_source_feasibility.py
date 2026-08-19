import copy
from io import BytesIO
import json
import tarfile
import unittest
from pathlib import Path

from v124_sgd_source_feasibility import build_inventory, service_domain


def fixture_archive(config):
    root = f"dstc8-schema-guided-dialogue-{config['revision']}"
    buffer = BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        def add(name, value, raw=False):
            payload = value.encode() if raw else json.dumps(value).encode()
            info = tarfile.TarInfo(f"{root}/{name}")
            info.size = len(payload)
            archive.addfile(info, BytesIO(payload))

        add("LICENSE.txt", "Creative Commons Attribution-ShareAlike 4.0 International", raw=True)
        schemas = {
            "train": [{"service_name": "Seen_1", "intents": [{"name": "DoSeen"}]}],
            "dev": [{"service_name": "Seen_1", "intents": [{"name": "DoSeen"}]}],
            "test": [
                {"service_name": "Seen_1", "intents": [{"name": "DoSeen"}]},
                {"service_name": "Seen_2", "intents": [{"name": "DoNovel"}]},
                {"service_name": "Outside_1", "intents": [{"name": "DoOutside"}]},
            ],
        }
        for partition, schema in schemas.items():
            add(f"{partition}/schema.json", schema)
            services = [(row["service_name"], row["intents"][0]["name"]) for row in schema]
            dialogues = []
            for index, (service, intent) in enumerate(services):
                dialogues.append({
                    "dialogue_id": f"{partition}-{index}",
                    "turns": [{
                        "speaker": "USER", "utterance": "fixture language",
                        "frames": [{
                            "service": service,
                            "state": {"active_intent": intent},
                            "actions": [{"act": "INFORM_INTENT", "slot": "intent", "values": [intent]}],
                        }],
                    }],
                })
            add(f"{partition}/dialogues_001.json", dialogues)
    return buffer.getvalue()


class V124SourceTests(unittest.TestCase):
    def test_text_free_open_set_inventory(self):
        config = json.loads(Path("configs/v124-sgd-source-feasibility.json").read_text())
        local = copy.deepcopy(config)
        gates = local["sourceGates"]
        for key in (
            "minimumDialogueCount", "minimumDomainCount", "minimumServiceCount", "minimumIntentCount",
            "minimumIntentIntroductionCandidateCount", "minimumTrainCandidateCount",
            "minimumTestCandidateCountPerOpenSetClass", "minimumKnownTestDomainCoverage",
            "minimumNovelValidTestDomainCoverage", "minimumUnsupportedTestDomainCoverage",
        ):
            gates[key] = 1
        result = build_inventory(fixture_archive(local), local)
        self.assertTrue(result["source_pass"])
        self.assertEqual(result["test_open_set_class_counts"], {"known": 1, "novel_valid": 1, "unsupported": 1})
        self.assertFalse(result["contains_language_or_slot_values"])
        self.assertNotIn("utterance", result["candidate_index"][0])
        self.assertEqual(service_domain("RentalCars_12"), "rentalcars")


if __name__ == "__main__":
    unittest.main()
