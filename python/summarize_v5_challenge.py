#!/usr/bin/env python3
"""Write the locked V5 shortcut-challenge result report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", default="outputs/v5-challenge/frozen-probe/result.json")
    parser.add_argument("--output", default="docs/v5-challenge-results.md")
    return parser.parse_args()


def percent(value: float) -> str:
    return f"{value:.2%}"


def markdown(result: dict[str, Any]) -> str:
    decision = "passes" if result["gates"]["passed"] else "fails"
    gate_rows = [
        "| {name} | {value} | {minimum} | {status} |".format(
            name=check["name"],
            value=percent(check["value"]),
            minimum=percent(check["minimum"]),
            status="pass" if check["passed"] else "fail",
        )
        for check in result["gates"]["checks"]
    ]
    surface_rows = [
        f"| {name} | {percent(value['balanced_accuracy'])} | {value['roc_auc']:.3f} |"
        for name, value in result["by_surface"].items()
    ]
    mechanic_rows = [
        f"| {name} | {percent(value['balanced_accuracy'])} | {value['roc_auc']:.3f} |"
        for name, value in result["by_mechanic_canonical"].items()
    ]
    interval = result["canonical_grouped_bootstrap"][
        "balanced_accuracy_95_percentile_interval"
    ]
    renamed = result["surface_invariance"]["transformations"]["entity_renamed"]
    paraphrased = result["surface_invariance"]["transformations"]["paraphrased"]
    evidence = result["evidence_contrasts"]
    return "\n".join(
        [
            "# V5 locked shortcut-challenge results",
            "",
            "## Decision",
            "",
            f"The preregistered frozen-probe challenge **{decision}**. The locked canonical probe "
            f"reaches {percent(result['canonical']['balanced_accuracy'])} balanced accuracy and "
            f"{result['canonical']['roc_auc']:.3f} AUC on new simulator worlds. Its context-group "
            f"bootstrap interval is {percent(interval[0])}–{percent(interval[1])}.",
            "",
            "## Preregistered gates",
            "",
            "| Gate | Observed | Minimum | Result |",
            "| --- | ---: | ---: | --- |",
            *gate_rows,
            "",
            "## Surface robustness",
            "",
            "| Surface | Balanced accuracy | AUC |",
            "| --- | ---: | ---: |",
            *surface_rows,
            "",
            f"Canonical/entity-renamed prediction agreement is {percent(renamed['prediction_agreement'])}; "
            f"canonical/paraphrased agreement is {percent(paraphrased['prediction_agreement'])}. "
            f"All three surfaces are simultaneously correct for "
            f"{percent(result['surface_invariance']['complete_triplet_accuracy'])} of base records.",
            "",
            "## Held-out mechanics",
            "",
            "| Mechanic | Canonical balanced accuracy | AUC |",
            "| --- | ---: | ---: |",
            *mechanic_rows,
            "",
            "## Evidence contrasts",
            "",
            f"The two simulator-derived evidence groups contain {evidence['cross_label_comparisons']} "
            f"cross-label comparisons. Directional accuracy is "
            f"{percent(evidence['directional_accuracy'])}, and complete group classification is "
            f"{percent(evidence['complete_group_accuracy'])}. Because both groups come from the "
            "short-start relock family, this remains a narrow diagnostic rather than a broad "
            "evidence-rung generalization claim.",
            "",
            "## Firewall",
            "",
            f"- Frozen lock SHA-256: `{result['lock_sha256']}`.",
            f"- Challenge dataset SHA-256: `{result['challenge_dataset_sha256']}`.",
            f"- Records / base records / context groups: {result['records']} / "
            f"{result['base_records']} / {result['context_groups']}.",
            f"- Truncated prompts: {result['truncated_prompts']}.",
            "- Challenge evaluations: 1.",
            "- V3 test records read: 0.",
            "",
        ]
    )


def main() -> None:
    args = parse_args()
    result = json.loads(Path(args.result).read_text())
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(markdown(result))
    print(json.dumps({"output": str(output), "gates_passed": result["gates"]["passed"]}))


if __name__ == "__main__":
    main()
