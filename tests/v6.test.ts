import { describe, expect, it } from 'vitest';
import type { AgentEpistemicInput, AgentEpistemicRecord, TransitionTarget } from '../src/contracts';
import { mergeObservationallyEquivalentRecords, transformV6Surface, v6CanonicalInput } from '../src/v6';

const outcome: TransitionTarget = {
  success: true,
  next_location: 'observatory',
  inventory_added: [],
  inventory_removed: [],
  flags_changed: {},
  visible_actions_added: [],
  visible_actions_removed: [],
  blocked_actions_added: [],
  blocked_actions_removed: [],
  hidden_actions_revealed: [],
  hidden_actions_concealed: [],
  reachable_room_delta: 0,
  environment_changed: false,
};

function input(history: string): AgentEpistemicInput {
  return {
    task: 'predict_possible_transitions',
    goal: 'restore the beacon before the hour repeats',
    observation: {
      turn: 1,
      location: 'observatory',
      locationName: 'Observatory',
      description: 'The beacon waits.',
      sensory: [],
      exits: [],
      visibleObjects: [],
      characters: [],
      inventory: [],
      beliefs: [],
      memories: [],
      pressure: 0,
      signal: 0,
    },
    recent_history: [{ action: 'wait', outcome: history }],
    candidate_action: { key: 'wait', label: 'wait' },
    available_actions: [{ key: 'wait', label: 'wait' }],
  };
}

function record(id: string, target: TransitionTarget, scenario: string): AgentEpistemicRecord {
  return {
    id,
    schema_version: 2,
    split: 'train',
    split_group: 'context',
    agent_input: input('same visible history'),
    target: { identifiable: true, possible_outcomes: [target] },
    empirical_support: [],
    source_record_count: 1,
    source_scenario_ids: [scenario],
  };
}

describe('V6 shortcut-resistant corpus', () => {
  it('merges hidden worlds before reducing supervision to a binary label', () => {
    const merged = mergeObservationallyEquivalentRecords([
      record('left', outcome, 'gen-behavior-forced-powertrip-9101-trap'),
      record(
        'right',
        { ...outcome, success: false },
        'gen-behavior-forced-powertrip-9102-control',
      ),
    ]);

    expect(merged).toHaveLength(1);
    expect(merged[0].target.identifiable).toBe(false);
    expect(merged[0].target.possible_outcomes).toHaveLength(2);
    expect(merged[0].source_scenario_ids).toHaveLength(2);
  });

  it('creates three changed but action-preserving surface views', () => {
    const canonical = v6CanonicalInput(input('The shard fits the mirrored socket.'));
    const renamed = transformV6Surface(canonical, 'entity_renamed');
    const paraphrased = transformV6Surface(canonical, 'paraphrased');

    expect(canonical.goal).not.toBe(input('x').goal);
    expect(renamed).not.toEqual(canonical);
    expect(paraphrased).not.toEqual(canonical);
    expect(renamed.candidate_action.key).toBe(canonical.candidate_action.key);
    expect(paraphrased.candidate_action.key).toBe(canonical.candidate_action.key);
  });
});
