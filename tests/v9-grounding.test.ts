import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { describe, expect, test } from 'vitest';
import type { V8StructuredRecord } from '../src/contracts';
import { buildV9GroundingRecords } from '../src/v9-grounding';
import { validateV9Grounding } from '../src/v9-validation';

function sourceRecords(): V8StructuredRecord[] {
  return ['train', 'calibration'].flatMap((split) =>
    readFileSync(resolve(`data/v8/records/${split}.jsonl`), 'utf8').trim().split('\n')
      .map((line) => JSON.parse(line) as V8StructuredRecord),
  );
}

describe('V9 natural-language grounding corpus', () => {
  const records = buildV9GroundingRecords(sourceRecords(), 'test-v8-sha');

  test('creates complete context, template, mechanic, and operator strata', () => {
    const validation = validateV9Grounding(records);
    expect(records).toHaveLength(2160);
    expect(validation.errors).toEqual([]);
    expect(validation.context_cross_split_overlaps).toBe(0);
    expect(Object.keys(validation.templates)).toHaveLength(4);
    expect(Object.keys(validation.mechanics)).toHaveLength(6);
    expect(Object.keys(validation.operators)).toHaveLength(2);
  });

  test('keeps target labels and determinant ids out of observation prose', () => {
    for (const record of records) {
      expect(record.agent_input.observation).not.toMatch(/\b(?:active|inactive|resolved|unresolved|identifiable|ambiguous)\b/i);
      for (const determinant of record.action_dependency_schema.transition_determinants) {
        expect(record.agent_input.observation.toLowerCase()).not.toContain(determinant.id.toLowerCase());
      }
    }
  });

  test('stores exact evidence spans for every determinant', () => {
    for (const record of records) {
      for (const target of record.target.determinant_grounding) {
        expect(record.agent_input.observation.slice(target.evidence_span.start, target.evidence_span.end))
          .toBe(target.evidence_span.text);
      }
    }
  });
});
