#!/usr/bin/env python3
"""Write the concise reproducible V10 result report."""

from __future__ import annotations

import json
from pathlib import Path
from statistics import mean
from typing import Any, Callable

from v10_protocol import file_sha256


RESULT_PATH = Path("outputs/v10-frozen/evaluation/result.json")
AUDIT_PATH = Path("outputs/v10-pre-model/shortcut-audit.json")
FEATURE_METADATA_PATH = Path("outputs/v10-frozen/features/metadata.json")
MANIFEST_PATH = Path("data/v10/manifest.json")
OUTPUT_PATH = Path("docs/v10-results.md")


def metric_summary(folds: dict[str, Any], getter: Callable[[dict[str, Any]], float]) -> tuple[float, float, str]:
    values = [(getter(value["overall"]), name) for name, value in folds.items()]
    minimum, name = min(values)
    return minimum, mean(value for value, _ in values), name


def fmt(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.3f}"


def pretty(name: str) -> str:
    return name.replace("_", " ").replace(":", " / ").title()


def main() -> None:
    result = json.loads(RESULT_PATH.read_text())
    audit = json.loads(AUDIT_PATH.read_text())
    metadata = json.loads(FEATURE_METADATA_PATH.read_text())
    manifest = json.loads(MANIFEST_PATH.read_text())
    primary_name = result["primary_representation"]
    primary = result["representations"][primary_name]
    lines = [
        "# V10 results: current-state polarity decomposition",
        "",
        "## Verdict",
        "",
        "V10 is a clean negative transfer result. The deterministic decomposition remains exact, and the frozen 0.8B model can solve the complete task in the context-disjoint fold. None of the three readouts robustly transfers current-state polarity across held-out language families, however. The preregistered NLI-final primary fails every hard-gate family and is not eligible for LoRA or final evaluation.",
        "",
        "The locked decision is to run a separately preregistered larger-frozen-model capacity diagnostic using the same corpus and targets. V10 itself ran no larger model and no adapter.",
        "",
        "## Corpus and pre-model audit",
        "",
        f"The locked corpus contains {sum(manifest['validation']['records'].values()):,} records from {sum(manifest['validation']['contexts'].values())} semantic contexts, six mechanics, six language families, three state lexicons, two operator families, and {manifest['validation']['intervention_groups']} intervention groups. It contains {manifest['validation']['current_hypothesis_pairs']:,} current-state hypothesis pairs and {manifest['validation']['unresolved_hypothesis_pairs']:,} unresolved pairs.",
        "",
        "Generation produced zero structural, span, hypothesis, relation, allowed-value derivation, symbolic, balance, duplicate, or split-overlap errors. Current active/inactive targets are exactly balanced inside every split-by-mechanic-by-template-by-lexicon cell.",
        "",
        "All pre-model shortcut gates passed:",
        "",
    ]
    for check in audit["gates"]["checks"]:
        lines.append(f"- {pretty(check['name'])}: {check['value']:.3f} (maximum {check['maximum']:.3f}).")
    linguistic = audit["audits"]["reported_linguistic_baselines"]
    lines.extend([
        "",
        "Report-only character baselines confirmed legitimate language signal: match balanced accuracy "
        f"{linguistic['context_match']['balanced_accuracy']:.3f}, current polarity accuracy "
        f"{linguistic['context_current_polarity']['accuracy']:.3f}, hypothesis-relation balanced accuracy "
        f"{linguistic['context_relation']['balanced_accuracy']:.3f}, and temporal accuracy "
        f"{linguistic['context_temporal']['accuracy']:.3f}.",
        "",
        "## Frozen extraction",
        "",
        f"The one authorized `{metadata['model']}` extraction encoded {metadata['unique_base_prompts']:,} unique determinant/evidence prompts and {metadata['unique_nli_prompts']:,} unique hypothesis-conditioned prompts, covering {metadata['pair_examples']:,} candidate pairs. Base prompts used {metadata['minimum_base_prompt_tokens']}–{metadata['maximum_base_prompt_tokens']} tokens, NLI prompts used {metadata['minimum_nli_prompt_tokens']}–{metadata['maximum_nli_prompt_tokens']}, evidence spans used {metadata['minimum_evidence_span_tokens']}–{metadata['maximum_evidence_span_tokens']}, and no prompt was truncated.",
        "",
        "The representation comparison used the same locked 24 folds. `nli_final` was primary; the two direct heads were diagnostics and could not replace it after observing results.",
        "",
        "| Representation | Context oracle polarity | Minimum fold oracle polarity | Minimum fold temporal | Minimum full allowed values | Minimum symbolic BA | Minimum flip pairs | Minimum complete groups |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ])
    for representation, folds in result["representations"].items():
        context = folds["context"]["overall"]
        oracle = lambda value: value["ablations"]["oracle_span_oracle_temporal"]["polarity_accuracy"]
        temporal = lambda value: value["temporal_accuracy_predicted_span"]
        allowed = lambda value: value["ablations"]["fully_predicted"]["allowed_values_accuracy"]
        symbolic = lambda value: value["ablations"]["fully_predicted"]["symbolic_identifiability"]["balanced_accuracy"]
        flips = lambda value: value["ablations"]["fully_predicted"]["complete_flip_pair_accuracy"]
        minimum_oracle, _, _ = metric_summary(folds, oracle)
        minimum_temporal, _, _ = metric_summary(folds, temporal)
        minimum_allowed, _, _ = metric_summary(folds, allowed)
        minimum_symbolic, _, _ = metric_summary(folds, symbolic)
        minimum_flips, _, _ = metric_summary(folds, flips)
        minimum_groups = min(value["group_scope"]["complete_intervention_group_accuracy"] for value in folds.values())
        lines.append(
            f"| `{representation}` | {oracle(context):.3f} | {minimum_oracle:.3f} | {minimum_temporal:.3f} | "
            f"{minimum_allowed:.3f} | {minimum_symbolic:.3f} | {minimum_flips:.3f} | {minimum_groups:.3f} |"
        )

    lines.extend([
        "",
        "## Primary NLI-final folds",
        "",
        "| Fold | Span | Temporal | Oracle polarity | Pair consistency | Full allowed values | Symbolic BA | Flip pairs | Complete groups |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ])
    for name, fold in primary.items():
        overall = fold["overall"]
        oracle = overall["ablations"]["oracle_span_oracle_temporal"]
        full = overall["ablations"]["fully_predicted"]
        lines.append(
            f"| {pretty(name)} | {overall['span_accuracy']:.3f} | {overall['temporal_accuracy_predicted_span']:.3f} | "
            f"{oracle['polarity_accuracy']:.3f} | {oracle['hypothesis_pair_consistency']:.3f} | "
            f"{full['allowed_values_accuracy']:.3f} | {full['symbolic_identifiability']['balanced_accuracy']:.3f} | "
            f"{fmt(full['complete_flip_pair_accuracy'])} | {fmt(fold['group_scope']['complete_intervention_group_accuracy'])} |"
        )

    surface_candidates: dict[str, list[tuple[float, str]]] = {
        "span": [], "temporal": [], "oracle polarity": [], "pair consistency": [], "allowed values": [], "symbolic BA": [],
    }
    for fold_name, fold in primary.items():
        for surface, cell in fold["by_surface"].items():
            label = f"{fold_name} / {surface}"
            oracle = cell["ablations"]["oracle_span_oracle_temporal"]
            full = cell["ablations"]["fully_predicted"]
            surface_candidates["span"].append((cell["span_accuracy"], label))
            surface_candidates["temporal"].append((cell["temporal_accuracy_predicted_span"], label))
            surface_candidates["oracle polarity"].append((oracle["polarity_accuracy"], label))
            surface_candidates["pair consistency"].append((oracle["hypothesis_pair_consistency"], label))
            surface_candidates["allowed values"].append((full["allowed_values_accuracy"], label))
            surface_candidates["symbolic BA"].append((full["symbolic_identifiability"]["balanced_accuracy"], label))
    lines.extend(["", "Worst primary surface cells:", ""])
    for metric_name, values in surface_candidates.items():
        value, label = min(values)
        lines.append(f"- {metric_name}: {value:.3f} at `{label}`.")

    failed = [value for value in result["primary_gates"]["checks"] if not value["passed"]]
    lines.extend([
        "",
        "## Interpretation",
        "",
        "The context fold is the crucial control: NLI-final reaches 1.000 oracle polarity, pair consistency, allowed-value accuracy, symbolic balanced accuracy, and flip-pair accuracy. The pipeline and labels are therefore internally learnable. Its collapse under held-out templates is a transfer failure rather than a broken symbolic rule or impossible target.",
        "",
        "The direct diagnostics localize the same issue from another angle. Mean-direct retains stronger temporal transfer than evidence-span pooling, while evidence-span-direct reaches perfect context polarity. Yet their minimum held-out-template oracle polarity is only 0.477 and 0.170 respectively. A direct head learns whether the mentioned phrase correlates with a state inside known constructions, but does not reliably invert that meaning when assertion and negation operators change.",
        "",
        "The hypothesis-conditioned final-token interface does not fix this at 0.8B layer 6. It overfits the observed construction families: minimum oracle polarity and pair consistency are both 0.000, even with the gold evidence span and gold temporal status. Span and temporal cascades make the fully predicted ledger worse, but they are not the root cause of the primary polarity failure.",
        "",
        "The symbolic evaluator remains exact on all 3,240 records. Its robustness cannot rescue a grounding pipeline that frequently emits unresolved relation pairs: worst-fold symbolic balanced accuracy falls to approximately chance.",
        "",
        "## Gate decision",
        "",
        f"The primary failed {len(failed)} of {len(result['primary_gates']['checks'])} hard checks. The locked decision is `{result['decision']}`.",
        "",
        "Next, use a separate preregistration to compare the identical primary prompts and folds with frozen 4B and 9B representations. Do not change the corpus, templates, gates, layer-selection rule, or head after seeing V10. If a larger frozen representation restores oracle polarity and pair consistency, only then consider whether a small linguistic LoRA is useful for cost reduction. V10 authorizes neither LoRA nor final-mechanic access.",
        "",
        "## Reproducibility and firewall",
        "",
        f"- dataset: `{manifest['dataset_sha256']}`;",
        f"- pre-model audit: `{file_sha256(AUDIT_PATH)}`;",
        f"- frozen features: `{metadata['feature_artifact_sha256']}`;",
        f"- evaluation result: `{file_sha256(RESULT_PATH)}`.",
        "",
        "No larger frozen model, adapter, final mechanic, Tone Drift, V3 test record, prior holdout, untouched V8 mechanic, or V7 model result was accessed.",
    ])
    OUTPUT_PATH.write_text("\n".join(lines) + "\n")
    print(f"wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
