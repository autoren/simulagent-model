"""Render the audited V19 result report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", default="outputs/v19-frozen-integration/evaluation/result.json")
    parser.add_argument("--audit", default="outputs/v19-frozen-integration/post-result-audit.json")
    parser.add_argument("--correction", default="outputs/v19-frozen-integration/error-conditioning-replay.json")
    parser.add_argument("--output", default="docs/v19-results.md")
    args = parser.parse_args()
    result = json.loads((PROJECT_ROOT / args.result).read_text())
    audit = json.loads((PROJECT_ROOT / args.audit).read_text())
    correction = json.loads((PROJECT_ROOT / args.correction).read_text())
    if not audit["passed"]:
        raise ValueError("V19 post-result audit must pass before summarization")
    if not correction["passed"] or correction["primary_metrics_affected"]:
        raise ValueError("V19 grounding-error replay is not valid")
    supported = result["grounding"]["supported"]["development"]
    novel = result["grounding"]["novel_ontology"]["development"]
    lines = [
        "# V19 results: frozen grounding × executable schema induction",
        "",
        "The locked supported-language integration passes every preregistered gate. The frozen",
        "V15 grounder and unchanged V18 schema inducer complete all 40 development episodes exactly.",
        "This authorizes design of a fresh multi-mechanic final suite; it does not authorize LoRA.",
        "",
        f"Decision: `{result['decision']}`.",
        "",
        "## Grounding views",
        "",
        "| View | Role | Allowed values | Exact scenes | All-support episodes |",
        "|---|---|---:|---:|---:|",
        f"| `supported` | primary | {supported['allowed_value_accuracy']:.3f} | {supported['exact_scene_grounding']:.3f} | {supported['episodes_with_all_supports_exact']:.3f} |",
        f"| `novel_ontology` | diagnostic | {novel['allowed_value_accuracy']:.3f} | {novel['exact_scene_grounding']:.3f} | {novel['episodes_with_all_supports_exact']:.3f} |",
        "",
        "The primary view is perfect for active, inactive, and unresolved values. The diagnostic",
        f"view retains perfect span and temporal accuracy, but current polarity falls to {novel['current_polarity_accuracy']:.3f}; inactive-value accuracy is {novel['by_target_class']['inactive']['allowed_value_accuracy']:.3f}.",
        "",
        "## Supported-view two-by-two decomposition",
        "",
        "| Condition | Episode macro | Complete episodes | Target retained | Empty version |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, values in result["views"]["supported"]["conditions"].items():
        episode = values["episode_metrics"]
        schema = values["schema_recovery"]
        lines.append(
            f"| `{name}` | {episode['episode_macro_transition_set_exact_match']:.3f} | {episode['complete_episodes']}/{episode['episodes']} | {schema['target_retention_rate']:.3f} | {schema['empty_version_space_rate']:.3f} |"
        )
    lines.extend([
        "",
        "## Novel-ontology diagnostic",
        "",
        "| Condition | Episode macro | Complete episodes | Target retained | Empty version |",
        "|---|---:|---:|---:|---:|",
    ])
    for name, values in result["views"]["novel_ontology"]["conditions"].items():
        episode = values["episode_metrics"]
        schema = values["schema_recovery"]
        lines.append(
            f"| `{name}` | {episode['episode_macro_transition_set_exact_match']:.3f} | {episode['complete_episodes']}/{episode['episodes']} | {schema['target_retention_rate']:.3f} | {schema['empty_version_space_rate']:.3f} |"
        )
    novel_full = result["views"]["novel_ontology"]["conditions"]["frozen_support_frozen_query"]
    novel_support = result["views"]["novel_ontology"]["conditions"]["frozen_support_oracle_query"]
    novel_query = result["views"]["novel_ontology"]["conditions"]["oracle_support_frozen_query"]
    corrected_novel = correction["views"]["novel_ontology"]["frozen_support_frozen_query"]
    lines.extend([
        "",
        "Support grounding is the larger novel-ontology failure mode: frozen-support/oracle-query",
        f"episode accuracy is {novel_support['episode_metrics']['episode_macro_transition_set_exact_match']:.3f}, versus {novel_query['episode_metrics']['episode_macro_transition_set_exact_match']:.3f} with oracle supports and frozen queries. The full diagnostic reaches {novel_full['episode_metrics']['episode_macro_transition_set_exact_match']:.3f}, retains the target behavior in {novel_full['schema_recovery']['target_retention_rate']:.3f} of episodes, and invokes the locked empty-version rule in {novel_full['schema_recovery']['empty_version_space_rate']:.3f}.",
        "",
        f"The scope-correct error replay finds 28 zero-support-error episodes, all perfect. The one episode with one support error scores {corrected_novel['conditioned_on_support_grounding_errors']['one']['episode_macro_transition_set_exact_match']:.3f}; the 11 episodes with multiple support errors average {corrected_novel['conditioned_on_support_grounding_errors']['multiple']['episode_macro_transition_set_exact_match']:.3f}, retain the target in {corrected_novel['conditioned_on_support_grounding_errors']['multiple']['target_retention_rate']:.3f}, and produce an empty version space in {corrected_novel['conditioned_on_support_grounding_errors']['multiple']['empty_version_space_rate']:.3f}.",
        "",
        "## Development axes in the novel diagnostic",
        "",
        "| Axis | Episode macro | Complete episodes | Target retained | Empty version |",
        "|---|---:|---:|---:|---:|",
    ])
    for axis, values in novel_full["by_axis"].items():
        lines.append(
            f"| `{axis}` | {values['episode_macro_transition_set_exact_match']:.3f} | {values['complete_episodes']}/{values['episodes']} | {values['target_retention_rate']:.3f} | {values['empty_version_space_rate']:.3f} |"
        )
    lines.extend([
        "",
        "The V18 `determinant_vocabulary` category is 8/8 here, while other fresh lexicons expose",
        "polarity errors. This is a diagnostic observation, not a tuned vocabulary selection.",
        "",
        "## Reproducibility and firewall",
        "",
        "- 6,912 grounding scenes over two exactly paired views;",
        "- 240 unique base prompts and 480 unique NLI prompts;",
        "- one locked extraction with 720 model forward passes and no truncation;",
        "- one locked integration evaluation;",
        "- frozen deployment heads reproduced bit-for-bit from V15-only development features;",
        "- no adapter training, head selection/refitting, target-guided repair, support deletion, or DSL expansion;",
        "- zero V17 record or model-result reads; and",
        f"- post-result replay reproduces all {audit['grounding_predictions']:,} saved grounding predictions and eight integration condition reports.",
        "",
        "The locked evaluator's non-gating grounding-error histograms initially paired determinant",
        "lists by serialization order. A zero-forward-pass scope-correct replay joins by determinant",
        "id and supersedes only those three diagnostic fields. Grounding predictions, schema search,",
        "query answers, gates, and the primary decision are unchanged.",
        "",
        "## Next decision",
        "",
        "Freeze the design of a genuinely fresh multi-mechanic final suite with mechanics as the",
        "sampling unit. It should include one- and two-bit outcomes, injective and non-injective",
        "tables, multiple depths, and read-only plus state-changing actions. The supported-language",
        "view is the primary preregistered condition; novel ontology should remain a separately",
        "reported transfer diagnostic. V17 remains exposed and cannot be reused.",
        "",
    ])
    (PROJECT_ROOT / args.output).write_text("\n".join(lines))


if __name__ == "__main__":
    main()
