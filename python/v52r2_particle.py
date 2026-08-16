"""V52r2: V52 particle inference with 100-digit final-joint assembly."""
from __future__ import annotations

from decimal import Decimal, localcontext

from v22_relational import canonical_json
from v52_particle import (
    _logsumexp,
    _normalize_log_weights,
    configuration_distribution,
    particle_filter_episode,
    particle_suffix_predictive,
    probability_marginal,
)


def particle_inference(
    registry,
    supports,
    query,
    budget: int,
    base_seed: int,
    population: str,
    record_id: str,
    repeat: int,
    ess_threshold_fraction: float = 0.5,
    likelihood_power: int = 1,
    track_ancestry: bool = False,
):
    support_logs, support_diagnostics = [], []
    for program_index, mechanic in enumerate(registry):
        total = Decimal(0)
        program_diagnostics = []
        possible = True
        for episode, support in enumerate(supports):
            world = {
                row["atom"]: row["allowed_values"][0]
                for row in support["initial_state"]
            }
            log_likelihood, _, diagnostics = particle_filter_episode(
                mechanic["program"], support["entities"], world,
                support["actions"], support["observations"], budget, base_seed,
                (
                    population, record_id, program_index, budget, repeat,
                    f"support-{episode}",
                ),
                ess_threshold_fraction, track_ancestry,
            )
            program_diagnostics.append(diagnostics)
            if log_likelihood is None:
                possible = False
                break
            total += Decimal(likelihood_power) * log_likelihood
        support_logs.append(total if possible else None)
        support_diagnostics.append(program_diagnostics)
    support_program = _normalize_log_weights(support_logs)
    support_log_evidence = _logsumexp(support_logs) - Decimal(len(registry)).ln()

    query_logs, filters, query_diagnostics = [], [], []
    prefix = query["prefix_length"]
    for program_index, mechanic in enumerate(registry):
        if not support_program[program_index]:
            query_logs.append(None)
            filters.append([])
            query_diagnostics.append({"extinct": True, "ticks": []})
            continue
        world = {
            row["atom"]: row["allowed_values"][0]
            for row in query["initial_state"]
        }
        log_likelihood, groups, diagnostics = particle_filter_episode(
            mechanic["program"], query["entities"], world,
            query["actions"][:prefix], query["observations"], budget, base_seed,
            (population, record_id, program_index, budget, repeat, "query"),
            ess_threshold_fraction, track_ancestry,
        )
        filters.append(groups)
        query_diagnostics.append(diagnostics)
        query_logs.append(
            None if log_likelihood is None
            else support_program[program_index].ln()
            + Decimal(likelihood_power) * log_likelihood
        )
    query_program = _normalize_log_weights(query_logs)
    query_conditional_log_evidence = _logsumexp(query_logs)

    with localcontext() as context:
        context.prec = 100
        joint, metadata, configuration = {}, {}, {}
        for program_index, (mechanic, groups) in enumerate(
            zip(registry, filters, strict=True)
        ):
            for configuration_key, state_mass in configuration_distribution(groups).items():
                key = canonical_json({
                    "program": mechanic["key"], "configuration": configuration_key
                })
                mass = query_program[program_index] * state_mass
                joint[key] = mass
                configuration[configuration_key] = configuration.get(
                    configuration_key, Decimal(0)
                ) + mass
                metadata[key] = {
                    "program_index": program_index,
                    "program_key": mechanic["key"],
                    "program_ordinal": mechanic["program_ordinal"],
                    "probability_ordinal": mechanic["probability_ordinal"],
                    "configuration_key": configuration_key,
                }
        joint_total = sum(joint.values(), Decimal(0))
        joint = {key: value / joint_total for key, value in joint.items()}
        configuration = {}
        for key, mass in joint.items():
            configuration_key = metadata[key]["configuration_key"]
            configuration[configuration_key] = configuration.get(
                configuration_key, Decimal(0)
            ) + mass
        configuration_total = sum(configuration.values(), Decimal(0))
        configuration = {
            key: value / configuration_total for key, value in configuration.items()
        }

    suffix = particle_suffix_predictive(registry, query_program, query, filters)
    return {
        "support_program": support_program,
        "query_program": query_program,
        "probability": probability_marginal(registry, query_program),
        "joint": joint,
        "metadata": metadata,
        "configuration": configuration,
        "suffix": suffix,
        "support_log_evidence_by_program": support_logs,
        "query_log_weight_by_program": query_logs,
        "record_log_evidence": support_log_evidence + query_conditional_log_evidence,
        "support_diagnostics": support_diagnostics,
        "query_diagnostics": query_diagnostics,
    }
