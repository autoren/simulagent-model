#!/usr/bin/env python3
"""Write the V16 scope-correct replay result."""

import json
from pathlib import Path

from v10_protocol import file_sha256


def main():
    result_path = Path("outputs/v16-scope-correct-replay/result.json")
    output_path = Path("docs/v16-results.md")
    result = json.loads(result_path.read_text())
    group = result["scope_correct_gates"]["checks"][-1]
    output_path.write_text(f"""# V16 results: scope-correct V15 gate replay

V16 passes all fourteen scope-correct gates without fitting a model, running inference, recomputing a prediction, changing a threshold, or accessing new data.

The complete-intervention-group gate applies to 15 transfer folds whose own evaluation masks contain complete six-record groups. Its worst value is {group['value']:.3f} on `{group['minimum_fold']}`, above the unchanged 0.50 threshold. Eleven lexicon/operator/combined folds correctly report the group metric as not applicable because their evaluation masks contain zero complete groups.

Decision: `{result['decision']}`.

This authorizes only the design of a separately locked final-mechanic evaluation using the unchanged frozen V15 architecture. No final mechanic has been accessed, and LoRA remains unauthorized.

- V16 lock: `{result['protocol_lock_sha256']}`;
- V16 result: `{file_sha256(result_path)}`;
- new fits / forward passes / predictions / threshold changes: 0 / 0 / 0 / 0.
""")
    print(f"wrote {output_path}")


if __name__ == "__main__":
    main()
