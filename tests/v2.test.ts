import { describe, expect, it } from 'vitest';
import { scenarioVariants } from '../../simulagent/src/simulation';
import { compileScenario } from '../src/dataset';
import { canonicalJson } from '../src/serialization';
import {
  buildAgentEpistemicRecords,
  balanceOutcomeCountTraining,
  buildPrivilegedV2Records,
  toOutcomeCountMlx,
  scenarioDynamicsSnapshot,
} from '../src/v2';
import { buildAgentEpistemicRecordsV3 } from '../src/v3';

const ratios = { train: 0.8, valid: 0.1, test: 0.1 };
const source = [
  ...compileScenario({
    scenario: scenarioVariants.baseline,
    split: 'train',
    maxStates: 3,
    maxDepth: 2,
  }),
  ...compileScenario({
    scenario: scenarioVariants['early-storm'],
    split: 'test',
    maxStates: 3,
    maxDepth: 2,
  }),
];

describe('dataset v2', () => {
  it('adds transition rules without scenario identity', () => {
    const baseline = scenarioDynamicsSnapshot(scenarioVariants.baseline);
    const early = scenarioDynamicsSnapshot(scenarioVariants['early-storm']);
    expect(baseline.storm_turn).not.toBe(early.storm_turn);
    expect(canonicalJson(baseline)).not.toContain('baseline');
  });

  it('deduplicates agent prompts into possible-outcome records', () => {
    const records = buildAgentEpistemicRecords({
      source,
      splitSeed: 'test-v2',
      splitRatios: ratios,
    });
    expect(new Set(records.map((record) => canonicalJson(record.agent_input))).size).toBe(
      records.length,
    );
    expect(records.some((record) => record.source_record_count > 1)).toBe(true);
    expect(
      records.every(
        (record) =>
          record.target.identifiable === (record.target.possible_outcomes.length === 1),
      ),
    ).toBe(true);
  });

  it('keeps all actions from one privileged context in one split', () => {
    const records = buildPrivilegedV2Records({
      source,
      splitSeed: 'test-v2',
      splitRatios: ratios,
    });
    const groupSplits = new Map<string, string>();
    for (const record of records) {
      const previous = groupSplits.get(record.split_group);
      expect(previous === undefined || previous === record.split).toBe(true);
      groupSplits.set(record.split_group, record.split);
    }
  });

  it('serializes a compact outcome-count calibration target', () => {
    const [record] = buildAgentEpistemicRecords({
      source,
      splitSeed: 'test-v2',
      splitRatios: ratios,
    });
    const example = toOutcomeCountMlx(record);
    const input = JSON.parse(example.messages[1].content) as { task: string };
    expect(input.task).toBe('count_possible_transitions');
    expect(example.messages[2].content).toBe(String(record.target.possible_outcomes.length));
  });

  it('balances identifiable and ambiguous count training examples', () => {
    const records = buildAgentEpistemicRecords({
      source,
      splitSeed: 'test-v2',
      splitRatios: ratios,
    });
    const identifiable = records[0];
    const ambiguous = {
      ...identifiable,
      id: `${identifiable.id}:ambiguous-fixture`,
      target: {
        identifiable: false,
        possible_outcomes: [
          identifiable.target.possible_outcomes[0],
          { ...identifiable.target.possible_outcomes[0], success: false },
        ],
      },
    };
    const balanced = balanceOutcomeCountTraining([
      identifiable,
      { ...identifiable, id: `${identifiable.id}:second-identifiable` },
      ambiguous,
    ]);
    expect(balanced.filter((record) => record.target.identifiable)).toHaveLength(
      balanced.filter((record) => !record.target.identifiable).length,
    );
  });
});

describe('dataset v3', () => {
  it('assigns whole observation contexts with mechanic metadata', () => {
    const built = buildAgentEpistemicRecordsV3({
      source,
      splitSeed: 'test-v3',
      splitRatios: ratios,
      stratificationRestarts: 4,
      minimumMechanicSupport: 1,
    });
    const contextSplits = new Map<string, string>();
    for (const record of built.records) {
      const previous = contextSplits.get(record.split_group);
      expect(previous === undefined || previous === record.split).toBe(true);
      contextSplits.set(record.split_group, record.split);
      expect(record.schema_version).toBe(3);
      expect(record.mechanic_labels.length).toBeGreaterThan(0);
    }
  });
});
