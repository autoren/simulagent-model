import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { describe, expect, test } from 'vitest';
import type { V8ActionDependencySchema, V8StructuredRecord } from '../src/contracts';
import { allowedValuesFromV8Input, evaluateAllowedTransitions } from '../src/v9-symbolic';

function schema(codes: string[]): V8ActionDependencySchema {
  return {
    candidate_action: 'test action',
    transition_determinants: [
      { id: 'a', label: 'factor a' },
      { id: 'b', label: 'factor b' },
    ],
    transition_cases: [
      { values: ['inactive', 'inactive'], transition_code: codes[0] },
      { values: ['active', 'inactive'], transition_code: codes[1] },
      { values: ['inactive', 'active'], transition_code: codes[2] },
      { values: ['active', 'active'], transition_code: codes[3] },
    ],
    rule: 'Only the listed determinant roles may change the transition for this action.',
  };
}

function readV8Records(): V8StructuredRecord[] {
  return ['train', 'calibration'].flatMap((split) =>
    readFileSync(resolve(`data/v8/records/${split}.jsonl`), 'utf8')
      .trim()
      .split('\n')
      .map((line) => JSON.parse(line) as V8StructuredRecord),
  );
}

describe('V9 symbolic evaluator', () => {
  test('enumerates compatible assignments and deduplicates transition codes', () => {
    const result = evaluateAllowedTransitions({
      action_dependency_schema: schema(['no', 'no', 'no', 'yes']),
      determinant_values: [
        { determinant_id: 'a', allowed_values: ['inactive', 'active'] },
        { determinant_id: 'b', allowed_values: ['active'] },
      ],
    });
    expect(result).toEqual({
      compatible_assignments: 2,
      possible_transition_codes: ['no', 'yes'],
      identifiable: false,
    });
  });

  test('rejects incomplete or malformed grounding instead of guessing', () => {
    expect(() => evaluateAllowedTransitions({
      action_dependency_schema: schema(['a', 'b', 'c', 'd']),
      determinant_values: [{ determinant_id: 'a', allowed_values: ['active'] }],
    })).toThrow(/omits determinant b/);
    expect(() => evaluateAllowedTransitions({
      action_dependency_schema: schema(['a', 'b', 'c', 'd']),
      determinant_values: [
        { determinant_id: 'a', allowed_values: [] },
        { determinant_id: 'b', allowed_values: ['active'] },
      ],
    })).toThrow(/no allowed values/);
  });

  test('exactly reproduces all 6,480 simulator-derived V8 oracle labels', () => {
    const records = readV8Records();
    expect(records).toHaveLength(6480);
    for (const record of records) {
      const result = evaluateAllowedTransitions({
        action_dependency_schema: record.agent_input.action_dependency_schema,
        determinant_values: allowedValuesFromV8Input(record.agent_input),
      });
      expect(result.identifiable, record.id).toBe(!record.target.ambiguous);
      expect(result.possible_transition_codes.length, record.id).toBe(record.target.possible_transition_count);
      expect(result.compatible_assignments, record.id).toBe(record.oracle.compatible_assignments);
    }
  });
});
