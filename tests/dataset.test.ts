import { describe, expect, it } from 'vitest';
import { createInitialState, scenarioVariants } from '../../simulagent/src/simulation';
import { compileScenario, stateFingerprint } from '../src/dataset';
import { toMlxExample } from '../src/mlx';
import { validateRecords } from '../src/validation';

describe('counterfactual dataset compiler', () => {
  it('emits every available action with an oracle delta', () => {
    const records = compileScenario({
      scenario: scenarioVariants.baseline,
      split: 'train',
      maxStates: 1,
      maxDepth: 0,
    });
    expect(records.length).toBeGreaterThan(1);
    expect(new Set(records.map((record) => record.state_id)).size).toBe(1);
    expect(records.every((record) => typeof record.target.success === 'boolean')).toBe(true);
    expect(records.every((record) => record.agent_input.task === 'predict_transition')).toBe(true);
  });

  it('does not place privileged state in the agent prompt', () => {
    const [record] = compileScenario({
      scenario: scenarioVariants['visible-key'],
      split: 'train',
      maxStates: 1,
      maxDepth: 0,
    });
    const agent = toMlxExample(record, 'agent');
    const privileged = toMlxExample(record, 'privileged');
    expect(agent.messages[1].content).not.toContain('privileged_world_state');
    expect(privileged.messages[1].content).toContain('privileged_world_state');
  });

  it('produces stable fingerprints and valid records', () => {
    const first = createInitialState('early-storm');
    const second = createInitialState('early-storm');
    expect(stateFingerprint(first)).toBe(stateFingerprint(second));

    const records = compileScenario({
      scenario: scenarioVariants['early-storm'],
      split: 'train',
      maxStates: 3,
      maxDepth: 2,
    });
    const summary = validateRecords(records, { requireAllSplits: false });
    expect(summary.recordCount).toBe(records.length);
  });
});
