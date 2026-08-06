import { describe, expect, it } from 'vitest';
import type { AgentEpistemicRecordV3 } from '../src/contracts';
import {
  balanceBinaryTraining,
  buildBinaryIdentifiabilityRecordsV4,
  toBinaryIdentifiabilityMlx,
} from '../src/v4';
import { validateV4 } from '../src/v4-validation';

function fixture(index: number, split: 'train' | 'valid'): AgentEpistemicRecordV3 {
  const identifiable = index % 2 === 0;
  const outcome = {
    success: true,
    next_location: 'room',
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
  return {
    id: `agent-v3:${split}:${index}`,
    schema_version: 3,
    split,
    split_group: `context-${split}-${Math.floor(index / 2)}`,
    agent_input: {
      task: 'predict_possible_transitions',
      goal: 'test',
      observation: {
        turn: index,
        location: 'atrium',
        locationName: 'Room',
        description: 'A room.',
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
      recent_history: [],
      candidate_action: { key: index % 3 === 0 ? 'wait' : 'inspect:room', label: 'act' },
      available_actions: [],
    },
    target: {
      identifiable,
      possible_outcomes: identifiable ? [outcome] : [outcome, { ...outcome, success: false }],
    },
    empirical_support: [],
    source_record_count: 1,
    source_scenario_ids: ['fixture'],
    mechanic_labels: [`family:${index % 2}`],
  };
}

describe('dataset v4 binary calibration track', () => {
  const built = buildBinaryIdentifiabilityRecordsV4({
    sourceTrain: Array.from({ length: 40 }, (_, index) => fixture(index, 'train')),
    sourceValidation: Array.from({ length: 10 }, (_, index) => fixture(index + 100, 'valid')),
    splitSeed: 'test-v4',
    calibrationRatio: 0.2,
    stratificationRestarts: 8,
  });

  it('keeps train, calibration, and validation context-disjoint', () => {
    const report = validateV4(built.records);
    expect(report.errors).toEqual([]);
    expect(report.context_cross_split_overlaps).toBe(0);
    expect(report.source_test_records_read).toBe(0);
    expect(report.counts.train).toBeGreaterThan(0);
    expect(report.counts.calibration).toBeGreaterThan(0);
    expect(report.counts.validation).toBe(10);
  });

  it('serializes single-token A/B labels and balances training', () => {
    const train = built.records.filter((record) => record.split === 'train');
    const balanced = balanceBinaryTraining(train);
    expect(balanced.filter((record) => record.target.identifiable)).toHaveLength(
      balanced.filter((record) => !record.target.identifiable).length,
    );
    const example = toBinaryIdentifiabilityMlx(balanced[0]);
    expect(['A', 'B']).toContain(example.messages[2].content);
    expect(JSON.parse(example.messages[1].content).task).toBe('classify_identifiability');
  });
});
