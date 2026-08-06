import type { ScenarioConfig } from '../../simulagent/src/simulation';
import { scenarioVariants, type ScenarioVariantId } from '../../simulagent/src/simulation';
import type {
  AgentEpistemicInput,
  AgentEpistemicRecord,
  AgentEpistemicRecordV3,
  AgentEpistemicTarget,
  AgentOutcomeCountInput,
  CounterfactualRecord,
  DatasetSplit,
  MlxExample,
  PrivilegedTransitionInputV2,
  PrivilegedTransitionRecordV2,
  ScenarioDynamicsSnapshot,
  SplitRatios,
  TransitionTarget,
} from './contracts';
import { transitionSystemPrompt } from './mlx';
import { canonicalJson, sha256, shortHash } from './serialization';
import { createSplitPlan } from './split';

export const epistemicSystemPrompt = [
  'Predict the complete set of transitions supported by observationally equivalent deterministic worlds.',
  'Use only the supplied observation history and candidate action.',
  'Do not choose another action and do not add explanation.',
  'Return exactly one JSON object with fields identifiable and possible_outcomes.',
  'identifiable is true only when possible_outcomes contains exactly one transition.',
  'Each possible outcome must contain exactly these fields: blocked_actions_added, blocked_actions_removed, environment_changed, flags_changed, hidden_actions_concealed, hidden_actions_revealed, inventory_added, inventory_removed, next_location, reachable_room_delta, success, visible_actions_added, visible_actions_removed.',
  'success and environment_changed are booleans; next_location is a string; reachable_room_delta is an integer; flags_changed is a JSON object; every other transition field is an array of strings.',
  'Do not include action names, narrative outcomes, next-state descriptions, or any other fields.',
].join(' ');

export const outcomeCountSystemPrompt = [
  'Count the distinct transitions supported by observationally equivalent deterministic worlds.',
  'Use only the supplied observation history and candidate action.',
  'Do not predict transition contents and do not add explanation.',
  'Return exactly one ASCII digit from 1 through 5 and nothing else.',
].join(' ');

export function scenarioDynamicsSnapshot(scenario: ScenarioConfig): ScenarioDynamicsSnapshot {
  return {
    storm_turn: scenario.stormTurn,
    generator_pressure_interval: scenario.generatorPressureInterval,
    hatch_relock_turn: scenario.hatchRelockTurn ?? null,
    hatch_relock_pressure: scenario.hatchRelockPressure ?? 1,
    generator_trip_after_mirror_install: scenario.generatorTripAfterMirrorInstall ?? false,
    announce_environment_changes: scenario.announceEnvironmentChanges ?? false,
    announce_hatch_relock_upstream: scenario.announceHatchRelockUpstream ?? false,
    announce_hatch_relock_consequence: scenario.announceHatchRelockConsequence ?? false,
    announce_hatch_relock_procedure: scenario.announceHatchRelockProcedure ?? false,
    announce_power_trip_consequence: scenario.announcePowerTripConsequence ?? false,
    announce_power_trip_procedure: scenario.announcePowerTripProcedure ?? false,
    hide_blocked_exit_signals: scenario.hideBlockedExitSignals ?? false,
    hide_power_state_signals: scenario.hidePowerStateSignals ?? false,
  };
}

export function buildAgentEpistemicRecords(options: {
  source: CounterfactualRecord[];
  splitSeed: string;
  splitRatios: SplitRatios;
}): AgentEpistemicRecord[] {
  const promptGroups = new Map<string, CounterfactualRecord[]>();
  for (const record of options.source) {
    const key = canonicalJson(toEpistemicInput(record));
    promptGroups.set(key, [...(promptGroups.get(key) ?? []), record]);
  }

  const contextGroups = [...promptGroups.values()].map((records) =>
    agentContextGroup(toEpistemicInput(records[0])),
  );
  const splitPlan = createSplitPlan(contextGroups, options.splitRatios, options.splitSeed);

  return [...promptGroups.values()]
    .map((records): AgentEpistemicRecord => {
      const input = toEpistemicInput(records[0]);
      const splitGroup = agentContextGroup(input);
      const split = requiredSplit(splitPlan, splitGroup);
      const targets = new Map<string, { target: TransitionTarget; count: number }>();
      for (const record of records) {
        const key = canonicalJson(record.target);
        const current = targets.get(key);
        targets.set(key, {
          target: record.target,
          count: (current?.count ?? 0) + 1,
        });
      }
      const outcomes = [...targets.entries()].sort(([left], [right]) => left.localeCompare(right));
      const target: AgentEpistemicTarget = {
        identifiable: outcomes.length === 1,
        possible_outcomes: outcomes.map(([, value]) => value.target),
      };
      return {
        id: `agent-v2:${shortHash(input, 24)}`,
        schema_version: 2,
        split,
        split_group: splitGroup,
        agent_input: input,
        target,
        empirical_support: outcomes.map(([key, value]) => ({
          target_sha256: sha256(key),
          count: value.count,
        })),
        source_record_count: records.length,
        source_scenario_ids: [...new Set(records.map((record) => record.scenario_id))].sort(),
      };
    })
    .sort(recordOrder);
}

export function buildPrivilegedV2Records(options: {
  source: CounterfactualRecord[];
  splitSeed: string;
  splitRatios: SplitRatios;
}): PrivilegedTransitionRecordV2[] {
  const enriched = options.source.map((record) => {
    const scenario = scenarioVariants[record.scenario_id as ScenarioVariantId];
    if (!scenario) {
      throw new Error(`Unknown scenario ${record.scenario_id}.`);
    }
    const input: PrivilegedTransitionInputV2 = {
      ...record.privileged_input,
      transition_rules: scenarioDynamicsSnapshot(scenario),
    };
    return { record, input, splitGroup: privilegedContextGroup(input) };
  });
  const splitPlan = createSplitPlan(
    enriched.map((value) => value.splitGroup),
    options.splitRatios,
    `${options.splitSeed}:privileged`,
  );

  return enriched
    .map(({ record, input, splitGroup }): PrivilegedTransitionRecordV2 => ({
      id: `privileged-v2:${record.id}`,
      schema_version: 2,
      split: requiredSplit(splitPlan, splitGroup),
      split_group: splitGroup,
      source_split: record.split,
      source_split_group: record.split_group,
      scenario_id: record.scenario_id,
      scenario_family: record.scenario_family,
      state_id: record.state_id,
      action: record.action,
      agent_input: record.agent_input,
      privileged_input: input,
      target: record.target,
    }))
    .sort(recordOrder);
}

export function toAgentV2Mlx(record: AgentEpistemicRecord | AgentEpistemicRecordV3): MlxExample {
  return {
    messages: [
      { role: 'system', content: epistemicSystemPrompt },
      { role: 'user', content: canonicalJson(record.agent_input) },
      { role: 'assistant', content: canonicalJson(record.target) },
    ],
  };
}

export function toOutcomeCountMlx(
  record: AgentEpistemicRecord | AgentEpistemicRecordV3,
): MlxExample {
  const input: AgentOutcomeCountInput = {
    ...record.agent_input,
    task: 'count_possible_transitions',
  };
  return {
    messages: [
      { role: 'system', content: outcomeCountSystemPrompt },
      { role: 'user', content: canonicalJson(input) },
      {
        role: 'assistant',
        content: String(record.target.possible_outcomes.length),
      },
    ],
  };
}

export function balanceOutcomeCountTraining<
  T extends AgentEpistemicRecord | AgentEpistemicRecordV3,
>(records: T[]): T[] {
  const identifiable = records.filter((record) => record.target.identifiable);
  const ambiguous = records.filter((record) => !record.target.identifiable);
  if (identifiable.length === 0 || ambiguous.length === 0) {
    return [...records];
  }
  const targetSize = Math.max(identifiable.length, ambiguous.length);
  const repeatTo = (values: T[]): T[] =>
    Array.from({ length: targetSize }, (_, index) => values[index % values.length]);
  return [...repeatTo(identifiable), ...repeatTo(ambiguous)];
}

export function toPrivilegedV2Mlx(record: PrivilegedTransitionRecordV2): MlxExample {
  return {
    messages: [
      { role: 'system', content: transitionSystemPrompt },
      { role: 'user', content: canonicalJson(record.privileged_input) },
      { role: 'assistant', content: canonicalJson(record.target) },
    ],
  };
}

function toEpistemicInput(record: CounterfactualRecord): AgentEpistemicInput {
  const { task: _task, ...input } = record.agent_input;
  return { ...input, task: 'predict_possible_transitions' };
}

function agentContextGroup(input: AgentEpistemicInput): string {
  const { candidate_action: _candidate, ...context } = input;
  return `agent-context:${shortHash(context, 24)}`;
}

function privilegedContextGroup(input: PrivilegedTransitionInputV2): string {
  const { candidate_action: _candidate, ...context } = input;
  return `privileged-context:${shortHash(context, 24)}`;
}

function requiredSplit(
  plan: Map<string, DatasetSplit>,
  group: string,
): DatasetSplit {
  const split = plan.get(group);
  if (!split) {
    throw new Error(`Missing split for ${group}.`);
  }
  return split;
}

function recordOrder(
  left: { split: DatasetSplit; split_group: string; id: string },
  right: { split: DatasetSplit; split_group: string; id: string },
): number {
  return canonicalJson([left.split, left.split_group, left.id]).localeCompare(
    canonicalJson([right.split, right.split_group, right.id]),
  );
}
