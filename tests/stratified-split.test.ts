import { describe, expect, it } from 'vitest';
import { createStratifiedSplitPlan } from '../src/stratified-split';

describe('stratified group splitting', () => {
  const groups = Array.from({ length: 40 }, (_, index) => ({
    id: `group-${index}`,
    features: {
      records: 4,
      [`class:${index % 2 ? 'ambiguous' : 'identifiable'}`]: 4,
      [`family:${index % 4}`]: 4,
    },
  }));

  it('is deterministic and preserves whole groups', () => {
    const left = createStratifiedSplitPlan(
      groups,
      { train: 0.8, valid: 0.1, test: 0.1 },
      'test-seed',
      8,
    );
    const right = createStratifiedSplitPlan(
      [...groups].reverse(),
      { train: 0.8, valid: 0.1, test: 0.1 },
      'test-seed',
      8,
    );
    expect([...left.plan].sort()).toEqual([...right.plan].sort());
    expect(new Set(left.plan.values())).toEqual(new Set(['train', 'valid', 'test']));
    expect(left.plan.size).toBe(groups.length);
  });

  it('keeps the balanced class rate aligned', () => {
    const result = createStratifiedSplitPlan(
      groups,
      { train: 0.8, valid: 0.1, test: 0.1 },
      'balanced-seed',
      8,
    );
    const rates = ['train', 'valid', 'test'].map((split) => {
      const selected = groups.filter((group) => result.plan.get(group.id) === split);
      return selected.filter((group) => group.features['class:ambiguous']).length / selected.length;
    });
    expect(Math.max(...rates) - Math.min(...rates)).toBeLessThanOrEqual(0.25);
  });
});
