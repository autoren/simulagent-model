import type {
  AgentEpistemicRecord,
  DatasetSplit,
  PrivilegedTransitionRecordV2,
  TransitionTarget,
} from './contracts';
import { canonicalJson, sha256 } from './serialization';

export interface V2ValidationSummary {
  agent: {
    records: number;
    counts: Record<DatasetSplit, number>;
    identifiable: number;
    ambiguous: number;
    prompt_cross_split_overlaps: number;
  };
  privileged: {
    records: number;
    counts: Record<DatasetSplit, number>;
    contradictory_prompts: number;
    prompt_cross_split_overlaps: number;
  };
}

export function validateV2(
  agent: AgentEpistemicRecord[],
  privileged: PrivilegedTransitionRecordV2[],
): V2ValidationSummary {
  const errors: string[] = [];
  const agentIds = new Set<string>();
  const agentPrompts = new Map<string, DatasetSplit>();
  const agentGroups = new Map<string, DatasetSplit>();

  for (const record of agent) {
    if (agentIds.has(record.id)) {
      errors.push(`Duplicate agent v2 id ${record.id}`);
    }
    agentIds.add(record.id);
    registerSplit(agentPrompts, canonicalJson(record.agent_input), record.split, errors, 'agent prompt');
    registerSplit(agentGroups, record.split_group, record.split, errors, 'agent context');
    if (record.target.possible_outcomes.length < 1) {
      errors.push(`No possible outcomes for ${record.id}`);
    }
    if (record.target.identifiable !== (record.target.possible_outcomes.length === 1)) {
      errors.push(`Incorrect identifiability flag for ${record.id}`);
    }
    if (record.empirical_support.length !== record.target.possible_outcomes.length) {
      errors.push(`Support/outcome length mismatch for ${record.id}`);
    }
    const support = record.empirical_support.reduce((sum, value) => sum + value.count, 0);
    if (support !== record.source_record_count) {
      errors.push(`Empirical support mismatch for ${record.id}`);
    }
    record.target.possible_outcomes.forEach((target, index) => {
      validateTarget(target, record.id, errors);
      if (record.empirical_support[index]?.target_sha256 !== sha256(canonicalJson(target))) {
        errors.push(`Support hash mismatch for ${record.id}`);
      }
    });
  }

  const privilegedIds = new Set<string>();
  const privilegedPrompts = new Map<string, { split: DatasetSplit; target: string }>();
  const privilegedGroups = new Map<string, DatasetSplit>();
  let contradictoryPrompts = 0;
  let privilegedCrossSplit = 0;
  for (const record of privileged) {
    if (privilegedIds.has(record.id)) {
      errors.push(`Duplicate privileged v2 id ${record.id}`);
    }
    privilegedIds.add(record.id);
    registerSplit(privilegedGroups, record.split_group, record.split, errors, 'privileged context');
    validateTarget(record.target, record.id, errors);
    const prompt = canonicalJson(record.privileged_input);
    const target = canonicalJson(record.target);
    const previous = privilegedPrompts.get(prompt);
    if (previous) {
      if (previous.split !== record.split) {
        privilegedCrossSplit += 1;
        errors.push(`Privileged prompt crosses ${previous.split}/${record.split}`);
      }
      if (previous.target !== target) {
        contradictoryPrompts += 1;
        errors.push(`Privileged Markov contradiction for ${record.id}`);
      }
    } else {
      privilegedPrompts.set(prompt, { split: record.split, target });
    }
  }

  requireSplits(agent, 'agent', errors);
  requireSplits(privileged, 'privileged', errors);
  if (errors.length > 0) {
    throw new Error(errors.slice(0, 30).join('\n'));
  }

  return {
    agent: {
      records: agent.length,
      counts: countSplits(agent),
      identifiable: agent.filter((record) => record.target.identifiable).length,
      ambiguous: agent.filter((record) => !record.target.identifiable).length,
      prompt_cross_split_overlaps: 0,
    },
    privileged: {
      records: privileged.length,
      counts: countSplits(privileged),
      contradictory_prompts: contradictoryPrompts,
      prompt_cross_split_overlaps: privilegedCrossSplit,
    },
  };
}

function registerSplit(
  values: Map<string, DatasetSplit>,
  key: string,
  split: DatasetSplit,
  errors: string[],
  label: string,
): void {
  const previous = values.get(key);
  if (previous && previous !== split) {
    errors.push(`${label} crosses ${previous}/${split}`);
  }
  values.set(key, split);
}

function requireSplits(
  records: Array<{ split: DatasetSplit }>,
  label: string,
  errors: string[],
): void {
  for (const split of ['train', 'valid', 'test'] as const) {
    if (!records.some((record) => record.split === split)) {
      errors.push(`${label} ${split} is empty`);
    }
  }
}

function countSplits(records: Array<{ split: DatasetSplit }>): Record<DatasetSplit, number> {
  return {
    train: records.filter((record) => record.split === 'train').length,
    valid: records.filter((record) => record.split === 'valid').length,
    test: records.filter((record) => record.split === 'test').length,
  };
}

function validateTarget(target: TransitionTarget, id: string, errors: string[]): void {
  if (typeof target.success !== 'boolean' || typeof target.environment_changed !== 'boolean') {
    errors.push(`Invalid target booleans for ${id}`);
  }
  if (
    typeof target.next_location !== 'string' ||
    !Number.isInteger(target.reachable_room_delta)
  ) {
    errors.push(`Invalid target scalars for ${id}`);
  }
  const arrays = [
    target.inventory_added,
    target.inventory_removed,
    target.visible_actions_added,
    target.visible_actions_removed,
    target.blocked_actions_added,
    target.blocked_actions_removed,
    target.hidden_actions_revealed,
    target.hidden_actions_concealed,
  ];
  if (
    arrays.some(
      (value) =>
        !Array.isArray(value) ||
        value.some((item) => typeof item !== 'string') ||
        new Set(value).size !== value.length,
    )
  ) {
    errors.push(`Invalid target arrays for ${id}`);
  }
  if (
    typeof target.flags_changed !== 'object' ||
    target.flags_changed === null ||
    Object.values(target.flags_changed).some((value) => typeof value !== 'boolean')
  ) {
    errors.push(`Invalid changed flags for ${id}`);
  }
}
