"""Render the immutable V20 development result."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", default="outputs/v20-probabilistic-interface/evaluation/result.json")
    parser.add_argument("--output", default="docs/v20-results.md")
    args = parser.parse_args()
    result = json.loads(Path(args.result).read_text())
    lines = [
        "# V20 results: calibrated grounding–program uncertainty",
        "",
        f"Decision: `{result['decision']}`.",
        "",
        "V20 reuses the saved V19 features and frozen deployment heads. It performs zero model",
        "forward passes, feature extractions, linear fits, adapter runs, or final-suite reads.",
        "",
        "## Calibration",
        "",
        "| View | Unique calibration prompts | Threshold | Calibration coverage | Development coverage | Development mean set size |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for view, values in result["views"].items():
        calibration = values["calibration"]
        development = values["development_coverage"]
        lines.append(
            f"| `{view}` | {calibration['unique_selected_current_prompts']} | "
            f"{calibration['threshold']:.4f} | {calibration['empirical_coverage']:.3f} | "
            f"{development['marginal_label_coverage']:.3f} | {development['mean_label_set_size']:.3f} |"
        )
    for view, values in result["views"].items():
        lines.extend([
            "",
            f"## {view.replace('_', ' ').title()}",
            "",
            "| Condition | Episode macro | Complete | Target retained | Empty | Excess outcomes/query |",
            "|---|---:|---:|---:|---:|---:|",
        ])
        for name, condition in values["conditions"].items():
            episode = condition["episode_metrics"]
            if name == "hard_support_hard_query":
                retention = condition["schema_recovery"]["target_retention_rate"]
                empty = condition["schema_recovery"]["empty_version_space_rate"]
                excess = "n/a"
            else:
                retention = condition["schema_recovery"]["target_credible_retention_rate"]
                empty = condition["schema_recovery"]["empty_posterior_rate"]
                excess = f"{condition['anti_widening']['mean_excess_outcomes']:.3f}"
            lines.append(
                f"| `{name}` | {episode['episode_macro_transition_set_exact_match']:.3f} | "
                f"{episode['complete_episodes']}/{episode['episodes']} | {retention:.3f} | "
                f"{empty:.3f} | {excess} |"
            )
    lines.extend([
        "",
        "## Preregistered checks",
        "",
    ])
    lines.extend(
        f"- `{name}`: {'pass' if passed else 'fail'}" for name, passed in result["checks"].items()
    )
    lines.extend([
        "",
        "The hard V19 result is unchanged. V20 cannot authorize LoRA and is eligible for the",
        "sealed V21 suite only according to the preregistered supported-preservation decision.",
        "",
    ])
    Path(args.output).write_text("\n".join(lines))


if __name__ == "__main__":
    main()
