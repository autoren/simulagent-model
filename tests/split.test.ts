import { describe, expect, it } from 'vitest';
import { createSplitPlan, splitGroupForScenario } from '../src/split';

describe('split grouping', () => {
  it('keeps authored trap/control pairs together', () => {
    expect(splitGroupForScenario('forced-relock-behavior-trap')).toBe(
      splitGroupForScenario('forced-relock-behavior-control'),
    );
    expect(splitGroupForScenario('locked-access-behavior-trap')).toBe(
      splitGroupForScenario('open-access-behavior-control'),
    );
  });

  it('keeps generated variants from one mechanic and seed together', () => {
    expect(splitGroupForScenario('gen-behavior-forced-relock-8001-trap')).toBe(
      splitGroupForScenario('gen-behavior-announced-relock-8001-control'),
    );
  });

  it('creates deterministic non-empty splits', () => {
    const groups = Array.from({ length: 20 }, (_, index) => `group-${index}`);
    const left = createSplitPlan(groups, { train: 0.8, valid: 0.1, test: 0.1 }, 'seed');
    const right = createSplitPlan([...groups].reverse(), { train: 0.8, valid: 0.1, test: 0.1 }, 'seed');
    expect([...left.entries()].sort()).toEqual([...right.entries()].sort());
    expect(new Set(left.values())).toEqual(new Set(['train', 'valid', 'test']));
  });
});

