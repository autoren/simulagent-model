"""Render the sealed V21 result and replay audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def row(name: str, value: dict, probabilistic: bool = False) -> str:
    episode = value["episode_metrics"]
    schema = value["schema_recovery"]
    retention = schema[
        "target_credible_retention_rate" if probabilistic else "target_retention_rate"
    ]
    empty = schema["empty_posterior_rate" if probabilistic else "empty_version_space_rate"]
    excess = value["anti_widening"]["mean_excess_outcomes"] if probabilistic else None
    return (
        f"| `{name}` | {episode['episode_macro_transition_set_exact_match']:.3f} | "
        f"{episode['complete_episodes']}/{episode['episodes']} | {retention:.3f} | "
        f"{empty:.3f} | {'n/a' if excess is None else f'{excess:.3f}'} |"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", default="outputs/v21-final/evaluation/result.json")
    parser.add_argument("--audit", default="outputs/v21-final/post-result-audit.json")
    parser.add_argument("--output", default="docs/v21-results.md")
    args = parser.parse_args()
    result = json.loads(Path(args.result).read_text())
    audit = json.loads(Path(args.audit).read_text())
    supported = result["views"]["supported"]
    novel = result["views"]["novel_ontology"]
    lines = [
        "# V21 results: sealed population multi-mechanic final",
        "",
        f"Decision: `{result['decision']}`.",
        "",
        "The hard V19-derived pipeline is the primary confirmatory system. The calibrated V20",
        "interface is a separately preregistered challenger and was already a negative ontology",
        "repair result before this suite was materialized.",
        "",
        "## Supported-language final",
        "",
        "| System | Mechanic macro | Complete mechanics | Target retained | Empty | Excess outcomes/query |",
        "|---|---:|---:|---:|---:|---:|",
        row("hard", supported["hard_conditions"]["frozen_support_frozen_query"]),
        row("probabilistic", supported["probabilistic_full"], True),
        "",
        f"The exact two-sided 95% interval for the hard system's complete-mechanic fraction is "
        f"[{result['complete_mechanic_exact_95_interval'][0]:.3f}, "
        f"{result['complete_mechanic_exact_95_interval'][1]:.3f}]. This interval is descriptive for "
        "the declared stratified mixture, not arbitrary mechanics.",
        "",
        "### Hard system by construction family",
        "",
        "| Family | Mechanic macro | Complete | Target retained | Empty |",
        "|---|---:|---:|---:|---:|",
    ]
    for family, value in supported["hard_conditions"]["frozen_support_frozen_query"]["by_axis"].items():
        lines.append(
            f"| `{family}` | {value['episode_macro_transition_set_exact_match']:.3f} | "
            f"{value['complete_episodes']}/{value['episodes']} | "
            f"{value['target_retention_rate']:.3f} | {value['empty_version_space_rate']:.3f} |"
        )
    lines.extend([
        "",
        "## Paired novel-ontology diagnostic",
        "",
        "| System | Mechanic macro | Complete mechanics | Target retained | Empty | Excess outcomes/query |",
        "|---|---:|---:|---:|---:|---:|",
        row("hard", novel["hard_conditions"]["frozen_support_frozen_query"]),
        row("probabilistic", novel["probabilistic_full"], True),
        "",
        "Novel ontology is non-gating and does not alter the supported-language decision.",
        "",
        "## Grounding by supported semantic operator",
        "",
        "| Operator | Hard allowed values | Polarity | Temporal | Span |",
        "|---|---:|---:|---:|---:|",
    ])
    for operator, value in supported["grounding"]["by_semantic_operator"].items():
        lines.append(
            f"| `{operator}` | {value['hard_allowed_value_accuracy']:.3f} | "
            f"{value['current_polarity_accuracy']:.3f} | {value['temporal_accuracy']:.3f} | "
            f"{value['span_accuracy']:.3f} |"
        )
    lines.extend([
        "",
        "## Firewall and reproducibility",
        "",
        "- the generator and metrics were locked before the final seed existed;",
        "- the V21r2 amendment changed only the prompt-inference budget from 2,000 to 5,200;",
        "- one delayed 256-bit seed, one construction, one extraction, and one evaluation;",
        "- no final labels used for fitting, selection, threshold changes, or retries;",
        f"- independent zero-forward-pass replay: {'pass' if audit['passed'] else 'fail'}; and",
        "- LoRA and final retries remain unauthorized.",
        "",
    ])
    Path(args.output).write_text("\n".join(lines))


if __name__ == "__main__":
    main()
